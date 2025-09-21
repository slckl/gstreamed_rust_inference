use std::io::Write;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use gstreamed_common::{discovery, pipeline::build_pipeline};
use gstreamer::{self as gst};
use gstreamer::{prelude::*, MessageView};
use image::{DynamicImage, RgbImage};
use inference_common::frame_meta::FrameMeta;
use inference_common::frame_times::{AggregatedTimes, FrameTimes};
use inference_common::img_dimensions::ImgDimensions;
use inference_common::tracker::similari::prelude::Sort;
use inference_common::video_meta::VideoMeta;
use ort::session::Session;

use crate::inference;

pub fn process_buffer(
    frame_dims: ImgDimensions,
    session: &mut Session,
    // TODO make tracking optional
    tracker: &Mutex<Sort>,
    agg_times: &mut AggregatedTimes,
    video_meta: &mut VideoMeta,
    buffer: &mut gst::Buffer,
) {
    let mut frame_times = FrameTimes::default();

    let start = Instant::now();
    // read buffer into an image
    let image = {
        let readable = buffer.map_readable().unwrap();
        let readable_vec = readable.to_vec();

        // buffer size is: width x height x 3
        let image = RgbImage::from_vec(
            frame_dims.width as u32,
            frame_dims.height as u32,
            readable_vec,
        )
        .unwrap();
        DynamicImage::ImageRgb8(image)
    };
    frame_times.frame_to_buffer = start.elapsed();

    // process it using some model + draw overlays on the output image
    let mut tracker = tracker.lock().unwrap();
    let (processed, bboxes) =
        inference::infer_on_image(session, Some(&mut *tracker), image, &mut frame_times).unwrap();
    let frame_meta = FrameMeta {
        pts: buffer.pts().unwrap_or_default().into(),
        dts: buffer.dts().unwrap_or_default().into(),
        bboxes_by_class: bboxes,
    };
    video_meta.push(frame_meta);

    // overwrite the buffer with our overlaid processed image
    let start = Instant::now();
    let buffer_mut = buffer.get_mut().unwrap();
    let mut writable = buffer_mut.map_writable().unwrap();
    let mut dst = writable.as_mut_slice();
    dst.write_all(processed.to_rgb8().as_raw()).unwrap();
    frame_times.buffer_to_frame = start.elapsed();

    log::debug!("{frame_times:?}");
    agg_times.push(frame_times);
}

/// Performs inference on a video file, using a gstreamer pipeline + ort.
pub fn process_video(input: &Path, live_playback: bool, session: Session) -> anyhow::Result<()> {
    gst::init()?;

    let agg_times = Arc::new(Mutex::new(AggregatedTimes::default()));

    // First, find out resolution of input file.
    log::info!("Discovering media properties of {input:?}");
    let file_info = discovery::discover(input)?;
    log::info!("{file_info:?}");
    let frame_dims = ImgDimensions::new(file_info.width as f32, file_info.height as f32);

    let output_path = input.with_extension("out.mkv");

    // Configure tracker, we use similari library, which provides iou/sort trackers.
    let tracker = inference_common::tracker::sort_tracker();

    // Build gst pipeline, which performs inference using the loaded model.
    let scoped_agg = Arc::clone(&agg_times);
    let video_meta = Arc::new(Mutex::new(VideoMeta::new(
        input.to_path_buf(),
        Some(output_path.clone()),
        frame_dims.width as u32,
        frame_dims.height as u32,
    )));
    let scoped_meta = Arc::clone(&video_meta);
    // FIXME can we do it without Mutex? it's not gonna be contested much, tho...
    let session = Arc::new(Mutex::new(session));
    let pipeline = build_pipeline(
        input.to_str().unwrap(),
        output_path.to_str().unwrap(),
        live_playback,
        move |buf| {
            let mut agg_times = scoped_agg.lock().unwrap();
            let mut video_meta = scoped_meta.lock().unwrap();
            let mut session = session.lock().unwrap();
            process_buffer(
                frame_dims,
                &mut session,
                &tracker,
                &mut agg_times,
                &mut video_meta,
                buf,
            );
        },
    )?;
    log::info!("Starting gst pipeline");

    // Make it play and listen to events to know when it's done.
    pipeline.set_state(gst::State::Playing).unwrap();

    let bus = pipeline.bus().unwrap();
    for msg in bus.iter_timed(gst::ClockTime::NONE) {
        match msg.view() {
            MessageView::Error(err) => {
                pipeline.debug_to_dot_file(gst::DebugGraphDetails::all(), "pipeline.error");
                let name = err.src().map(|e| e.name().to_string());
                log::error!("Error from element {name:?}: {}", err.error());
                break;
            }
            MessageView::Eos(..) => {
                log::info!("Pipeline reached end of stream.");
                break;
            }
            _ => (),
        }
    }

    let video_meta = video_meta.lock().unwrap();
    let output_json_path = input.with_extension("json");
    log::info!(
        "Writing output json file, {} frames: {output_json_path:?}",
        video_meta.frames.len()
    );
    serde_json::to_writer(std::fs::File::create(output_json_path)?, &*video_meta)?;

    pipeline.set_state(gst::State::Null).unwrap();

    // Print perf stats, ignoring first (outlier) frame.
    let agg = agg_times.lock().unwrap();
    let avg = agg.avg(true);
    log::info!("Average frame times: {avg:?}");

    let min = agg.min(true);
    log::info!("Min frame times: {min:?}");

    let max = agg.max(true);
    log::info!("Max frame times: {max:?}");

    Ok(())
}

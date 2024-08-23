use std::io::Write;
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

use gstreamed_common::frame_times::FrameTimes;
use gstreamed_common::{discovery, img_dimensions::ImgDimensions, pipeline::build_pipeline};
use gstreamed_tracker::similari::prelude::Sort;
use gstreamer::{self as gst};
use gstreamer::{prelude::*, MessageView};
use image::{DynamicImage, RgbImage};
use ort::Session;

use crate::inference;

pub fn process_buffer(
    frame_dims: ImgDimensions,
    session: &Session,
    // TODO make tracking optional
    tracker: &Mutex<Sort>,
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
    let processed =
        inference::infer_on_image(session, Some(&mut *tracker), image, &mut frame_times).unwrap();

    // overwrite the buffer with our overlaid processed image
    let start = Instant::now();
    let buffer_mut = buffer.get_mut().unwrap();
    let mut writable = buffer_mut.map_writable().unwrap();
    let mut dst = writable.as_mut_slice();
    dst.write_all(processed.to_rgb8().as_raw()).unwrap();
    frame_times.buffer_to_frame = start.elapsed();

    log::debug!("{frame_times:?}");
}

/// Performs inference on a video file, using a gstreamer pipeline + ort.
pub fn process_video(input: &Path, live_playback: bool, session: Session) -> anyhow::Result<()> {
    gst::init()?;

    // First, find out resolution of input file.
    let file_info = discovery::discover(input)?;
    log::info!("File info: {file_info:?}");
    let frame_dims = ImgDimensions::new(file_info.width as f32, file_info.height as f32);

    // Configure tracker, we use similari library, which provides iou/sort trackers.
    let tracker = gstreamed_tracker::sort_tracker();

    // Build gst pipeline, which performs inference using the loaded model.
    let pipeline = build_pipeline(input.to_str().unwrap(), live_playback, move |buf| {
        process_buffer(frame_dims, &session, &tracker, buf);
    })?;

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

    pipeline.set_state(gst::State::Null).unwrap();
    Ok(())
}

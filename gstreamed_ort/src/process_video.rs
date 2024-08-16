use std::io::Write;
use std::path::Path;
use std::time::Instant;

use gstreamed_common::frame_times::FrameTimes;
use gstreamed_common::{discovery, img_dimensions::ImgDimensions, pipeline::build_pipeline};
use gstreamer::{self as gst};
use gstreamer::{prelude::*, MessageView};
use image::{DynamicImage, RgbImage};
use ort::Session;

use crate::inference;

pub fn process_buffer(frame_dims: ImgDimensions, session: &Session, buffer: &mut gst::Buffer) {
    // let file_info = &self.file_info;
    // let model = &self.model;
    // let device = &self.device;

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
    let processed = inference::infer_on_image(session, image, &mut frame_times).unwrap();

    // processed.save("./output.jpg").unwrap();
    // std::process::exit(0);

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
pub fn process_video(path: &Path, session: Session) -> anyhow::Result<()> {
    gst::init()?;

    // First, find out resolution of input file.
    let file_info = discovery::discover(path)?;
    log::info!("File info: {file_info:?}");
    let frame_dims = ImgDimensions::new(file_info.width as f32, file_info.height as f32);

    // Build gst pipeline, which performs inference using the loaded model.
    let pipeline = build_pipeline(path.to_str().unwrap(), move |buf| {
        process_buffer(frame_dims, &session, buf);
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
                log::error!("Pipeline reached end of stream.");
                break;
            }
            _ => (),
        }
    }

    pipeline.set_state(gst::State::Null).unwrap();
    Ok(())
}

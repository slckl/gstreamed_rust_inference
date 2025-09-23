use std::path::Path;
use std::sync::{Arc, Mutex};

use candle_core::Device;
use gstreamed_common::{discovery, pipeline::build_pipeline};
use gstreamer::{self as gst};
use gstreamer::{prelude::*, MessageView};
use inference_common::frame_times::AggregatedTimes;
use inference_common::img_dimensions::ImgDimensions;

use crate::{inference, yolov8::YoloV8};

/// Performs inference on a video file, using a gstreamer pipeline + candle.
pub fn process_video(input: &Path, model: YoloV8, device: Device) -> anyhow::Result<()> {
    // First, find out resolution of input file.
    log::info!("Discovering media properties of {input:?}");
    let file_info = discovery::discover(input)?;
    log::info!("{file_info:?}");
    let frame_dims = ImgDimensions::new(file_info.width as f32, file_info.height as f32);

    let agg_times = Arc::new(Mutex::new(AggregatedTimes::default()));

    // Use tracker for candle pipeline, too.
    let tracker = inference_common::tracker::sort_tracker();

    let output_path = input.with_extension("out.mkv");

    // Build gst pipeline, which performs inference using the loaded model.
    let scoped_agg = Arc::clone(&agg_times);
    let pipeline = build_pipeline(
        input.to_str().unwrap(),
        output_path.to_str().unwrap(),
        false,
        move |buf| {
            let mut agg_times = scoped_agg.lock().unwrap();
            inference::process_buffer(frame_dims, &model, &device, &tracker, &mut agg_times, buf);
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

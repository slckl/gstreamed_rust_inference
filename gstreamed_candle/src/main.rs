mod inference;
mod yolov8;

use crate::inference::Which;
use candle_core::Device;
use clap::Parser;
use gstreamed_common::discovery;
use gstreamed_common::frame_times::AggregatedTimes;
use gstreamed_common::img_dimensions::ImgDimensions;
use gstreamed_common::pipeline::build_pipeline;
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer::MessageView;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Debug, Parser)]
pub struct Args {
    /// Path to a video file we want to process.
    input: PathBuf,
    #[arg(long, action, default_value = "false")]
    cuda: bool,
    // TODO dtype switch
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Initialize logging.
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    gst::init()?;

    // First, find out resolution of input file.
    let file_info = discovery::discover(&args.input)?;
    log::info!("{file_info:?}");
    let frame_dims = ImgDimensions::new(file_info.width as f32, file_info.height as f32);

    let device = if args.cuda {
        Device::new_cuda(0)?
    } else {
        Device::Cpu
    };

    // Load models using hf-hub.
    let which = Which::S;
    let model = inference::load_model(which, &device)?;

    let agg_times = Arc::new(Mutex::new(AggregatedTimes::default()));

    // Use tracker for candle pipeline, too.
    let tracker = gstreamed_tracker::sort_tracker();

    // Build gst pipeline, which performs inference using the loaded model.
    let scoped_agg = Arc::clone(&agg_times);
    let pipeline = build_pipeline(args.input.to_str().unwrap(), false, move |buf| {
        let mut agg_times = scoped_agg.lock().unwrap();
        inference::process_buffer(frame_dims, &model, &device, &tracker, &mut agg_times, buf);
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

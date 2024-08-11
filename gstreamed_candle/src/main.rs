mod discovery;
mod frame_times;
mod inference;
mod pipeline;
mod yolov8;

use crate::inference::Which;
use candle_core::Device;
use clap::Parser;
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer::MessageView;
use std::path::PathBuf;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use crate::pipeline::build_pipeline;

#[derive(Debug, Parser)]
pub struct Args {
    /// Path to a video file we want to process.
    input: PathBuf,
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

    // first find out resolution of input file
    let file_info = discovery::discover(&args.input)?;
    log::info!("File info: {file_info:?}");

    // let device = Device::Cpu;
    let device = Device::new_cuda(0)?;

    // load models
    let which = Which::S;
    let model = inference::load_model(which, &device)?;

    // TODO pipe gst logs to some rust log handler

    // build pipeline
    let pipeline = build_pipeline(args.input.to_str().unwrap(), file_info, model, device)?;

    // make it play and listen to events to know when it's done
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

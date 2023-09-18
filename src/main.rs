mod pipeline;
mod yolov8;

use clap::Parser;
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer::MessageView;

use crate::pipeline::build_pipeline;

#[derive(Debug, Parser)]
pub struct Args {
    /// Path to a video file we want to process.
    input: String,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    gst::init().unwrap();

    // TODO pipe gst logs to some rust log handler

    // build pipeline
    let pipeline = build_pipeline(&args.input)?;
    // make it play and listen to events to know when it's done
    pipeline.set_state(gst::State::Playing).unwrap();

    let bus = pipeline.bus().unwrap();
    for msg in bus.iter_timed(gst::ClockTime::NONE) {
        match msg.view() {
            MessageView::Error(err) => {
                let name = err.src().map(|e| e.name().to_string());
                eprintln!("Error from element {name:?}: {}", err.error());
                break;
            }
            MessageView::Eos(..) => {
                println!("Pipeline reached end of stream.");
                break;
            }
            _ => (),
        }
    }

    pipeline.set_state(gst::State::Null).unwrap();

    Ok(())
}

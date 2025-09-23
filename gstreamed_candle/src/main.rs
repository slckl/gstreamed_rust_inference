mod inference;
mod process_image;
mod process_video;
mod yolov8;

use crate::inference::Which;
use candle_core::Device;
use clap::Parser;
use gstreamer as gst;
use std::path::PathBuf;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Debug, Parser)]
pub struct Args {
    /// Path to input image (.jpeg/.png) or video file (.mp4/.mkv).
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

    let device = if args.cuda {
        Device::new_cuda(0)?
    } else {
        Device::Cpu
    };

    // Load models using hf-hub.
    let which = Which::S;
    let model = inference::load_model(which, &device)?;

    // Branch on file extension: video vs image.
    let ext = args
        .input
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase());

    match ext.as_deref() {
        Some("mp4") | Some("mkv") => process_video::process_video(&args.input, model, device)?,
        Some("jpeg") | Some("jpg") | Some("png") => {
            process_image::process_image(&args.input, model, device)?
        }
        Some(unk) => log::error!("Unhandled file extension: {unk}"),
        None => log::error!(
            "Input path does not have valid file extension: {:?}",
            args.input
        ),
    }

    Ok(())
}

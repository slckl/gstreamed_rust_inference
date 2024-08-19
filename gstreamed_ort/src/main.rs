mod inference;
mod process_image;
mod process_video;
mod yolo_parser;

use std::path::PathBuf;

use clap::Parser;
use ort::{CPUExecutionProvider, CUDAExecutionProvider, GraphOptimizationLevel, SessionBuilder};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Debug, Parser)]
pub struct Args {
    input: PathBuf,
    #[arg(long, action, default_value = "false")]
    cuda: bool,
    #[arg(long, short, default_value = "_models/yolov8s.onnx")]
    model: String,
}

fn main() -> anyhow::Result<()> {
    // Initialize logging.
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let args = Args::parse();

    // Load model into ort.
    let ep = if args.cuda {
        CUDAExecutionProvider::default().build()
        // ExecutionProvider::CUDA(Default::default())
    } else {
        CPUExecutionProvider::default().build()
        // ExecutionProvider::CPU(Default::default())
    };
    // TODO test trt exec provider, but requires a rebuild of onnxruntime with trt enabled
    // TODO warmup with synthetic image of the same dims

    ort::init().with_execution_providers([ep]).commit()?;

    let session = SessionBuilder::new()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        // .with_intra_threads(1)?
        .commit_from_file(args.model)?;
    // .with_model_from_file(args.model)?;
    log::debug!("session: {session:?}");

    match args.input.extension().and_then(|os_str| os_str.to_str()) {
        Some("mp4" | "mkv") => process_video::process_video(&args.input, session)?,
        Some("jpeg" | "jpg" | "png") => process_image::process_image(&args.input, &session)?,
        Some(unk) => log::error!("Unhandled file extension: {unk}"),
        None => log::error!(
            "Input path does not have valid file extension: {:?}",
            args.input
        ),
    }

    Ok(())
}

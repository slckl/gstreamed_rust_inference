mod inference;
mod yolo_parser;

use std::path::PathBuf;

use clap::Parser;
use ort::{ExecutionProvider, GraphOptimizationLevel, SessionBuilder};
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
        ExecutionProvider::CUDA(Default::default())
    } else {
        ExecutionProvider::CPU(Default::default())
    };
    // TODO test trt exec provider, but requires a rebuild of onnxruntime with trt enabled
    // TODO warmup with synthetic image of the same dims

    let ort_env = ort::Environment::builder()
        .with_name("yolov8")
        .with_execution_providers([ep])
        .build()?
        .into_arc();

    let session = SessionBuilder::new(&ort_env)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        // .with_intra_threads(1)?
        .with_model_from_file(args.model)?;
    log::debug!("session: {session:?}");

    // Read image.
    let og_image = image::open(args.input)?;

    // for _ in 0..10 {
    // Process image.
    let img = inference::process_image(&session, og_image.clone())?;
    // Save output.
    img.save("output.jpg")?;
    // }

    Ok(())
}

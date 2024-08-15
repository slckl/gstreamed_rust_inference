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

    // Load model into ort.
    // TODO warmup with synthetic image of the same dims
    let cuda_ep = ExecutionProvider::CUDA(Default::default());
    log::debug!("cuda.is_available(): {}", cuda_ep.is_available());
    let trt_ep = ExecutionProvider::TensorRT(Default::default());
    log::debug!("tensorrt.is_available(): {}", trt_ep.is_available());

    let ort_env = ort::Environment::builder()
        .with_name("yolov8")
        // .with_execution_providers([ExecutionProvider::CPU(Default::default())])
        // .with_execution_providers([cuda_ep])
        .with_execution_providers([trt_ep])
        .build()?
        .into_arc();

    let session = SessionBuilder::new(&ort_env)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        // .with_intra_threads(1)?
        .with_model_from_file("_models/yolov8s.640x360.cpu.onnx")?;
    log::debug!("session: {session:?}");

    // Read image.
    let og_image = image::open("sample.jpg")?;

    // Process image.
    let img = inference::process_image(&session, og_image)?;

    // Save output.
    img.save("output.jpg")?;

    Ok(())
}

mod inference;
mod yolo_parser;

use ort::{ExecutionProvider, GraphOptimizationLevel, SessionBuilder};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// Describes dimensions of an image.
#[derive(Debug, Copy, Clone)]
struct ImgDimensions {
    pub width: f32,
    pub height: f32,
}

impl ImgDimensions {
    pub fn new(width: f32, height: f32) -> Self {
        Self { width, height }
    }

    pub fn scale(&self, ratio: f32) -> Self {
        Self {
            width: self.width * ratio,
            height: self.height * ratio,
        }
    }
}

impl From<(u32, u32)> for ImgDimensions {
    fn from(value: (u32, u32)) -> Self {
        Self::new(value.0 as f32, value.1 as f32)
    }
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
    let ort_env = ort::Environment::builder()
        .with_name("yolov8")
        .with_execution_providers([ExecutionProvider::CPU(Default::default())])
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

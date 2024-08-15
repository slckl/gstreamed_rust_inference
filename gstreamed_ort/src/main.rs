mod inference;
mod yolo_parser;

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

    // Read image.
    let og_image = image::open("sample.jpg")?;

    // Process it.
    let img = inference::process_image(og_image)?;

    // Save output.
    img.save("output.jpg")?;

    Ok(())
}

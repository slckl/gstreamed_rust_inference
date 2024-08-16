/// Describes dimensions of an image.
#[derive(Debug, Copy, Clone)]
pub struct ImgDimensions {
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

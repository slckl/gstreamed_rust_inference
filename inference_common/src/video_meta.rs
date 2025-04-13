use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::frame_meta::FrameMeta;

/// Metadata corresponding to a processed video.
#[derive(Debug, Deserialize, Serialize)]
pub struct VideoMeta {
    /// Path to original input video file.
    pub input_file: PathBuf,
    pub width: u32,
    pub height: u32,
    /// Optional path to output video file, with inference overlays.
    pub output_file: Option<PathBuf>,
    /// Per-frame information with timestamps + recognized objects.
    pub frames: Vec<FrameMeta>,
}

impl VideoMeta {
    pub fn new(input_file: PathBuf, output_file: Option<PathBuf>, width: u32, height: u32) -> Self {
        Self {
            input_file,
            width,
            height,
            output_file,
            frames: Vec::new(),
        }
    }

    pub fn push(&mut self, frame: FrameMeta) {
        self.frames.push(frame);
    }
}

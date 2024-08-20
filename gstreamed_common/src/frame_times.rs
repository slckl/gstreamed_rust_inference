use std::fmt::{self, Debug};
use std::time::Duration;

/// Various timings for processing a single video frame.
#[derive(Default)]
pub struct FrameTimes {
    pub frame_to_buffer: Duration,
    pub buffer_resize: Duration,
    pub buffer_to_tensor: Duration,
    pub forward_pass: Duration,
    pub bbox_extraction: Duration,
    pub nms: Duration,
    pub tracking: Duration,
    pub annotation: Duration,
    pub buffer_to_frame: Duration,
}

impl FrameTimes {
    pub fn total(&self) -> Duration {
        self.frame_to_buffer
            + self.buffer_resize
            + self.buffer_to_tensor
            + self.forward_pass
            + self.bbox_extraction
            + self.nms
            + self.tracking
            + self.annotation
            + self.buffer_to_frame
    }
}

impl Debug for FrameTimes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "
            total: {:?},
            frame_to_buffer: {:?},
            buffer_resize: {:?},
            buffer_to_tensor: {:?},
            forward_pass: {:?},
            bbox_extraction: {:?},
            nms: {:?},
            tracking: {:?},
            annotation: {:?},
            buffer_to_frame: {:?}
            ",
            self.total(),
            self.frame_to_buffer,
            self.buffer_resize,
            self.buffer_to_tensor,
            self.forward_pass,
            self.bbox_extraction,
            self.nms,
            self.tracking,
            self.annotation,
            self.buffer_to_frame,
        )
    }
}

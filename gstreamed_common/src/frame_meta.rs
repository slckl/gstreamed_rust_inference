use serde::{Deserialize, Serialize};

use crate::bbox::BBoxesByClass;

#[derive(Debug, Serialize, Deserialize)]
pub struct FrameMeta {
    pub pts: u64,
    pub dts: u64,
    pub bboxes_by_class: BBoxesByClass,
}

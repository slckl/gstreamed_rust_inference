use std::path::Path;

use candle_core::Device;
use inference_common::{frame_meta::FrameMeta, frame_times::FrameTimes};

use crate::{inference, yolov8::YoloV8};

/// Performs inference on a single image file using Candle.
pub fn process_image(path: &Path, model: YoloV8, device: Device) -> anyhow::Result<()> {
    let mut frame_times = FrameTimes::default();

    // Read image.
    let og_image = image::open(path)?;

    // Process frame without tracking since it's meaningless for a single image
    let (annotated, bboxes) = inference::process_frame(
        og_image,
        &model,
        &device,
        None,
        0.25,
        0.45,
        14,
        &mut frame_times,
    )?;

    // Save output: annotated image & bboxes.
    let img_output_path = path.with_extension("out.jpg");
    annotated.save(img_output_path)?;

    let bbox_output_path = path.with_extension("out.json");
    let frame_meta = FrameMeta {
        pts: 0,
        dts: 0,
        bboxes_by_class: bboxes,
    };
    serde_json::to_writer(std::fs::File::create(bbox_output_path)?, &frame_meta)?;

    log::debug!("Frame times (single image): {frame_times:?}");

    Ok(())
}

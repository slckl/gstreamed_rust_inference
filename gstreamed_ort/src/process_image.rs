use std::path::Path;

use inference_common::{frame_meta::FrameMeta, frame_times::FrameTimes};
use ort::session::Session;

use crate::inference;

/// Performs inference on a single image file.
pub fn process_image(path: &Path, session: &Session) -> anyhow::Result<()> {
    let mut frame_times = FrameTimes::default();

    // Read image.
    let og_image = image::open(path)?;

    // Process image.
    let (img, bboxes) =
        inference::infer_on_image(session, None, og_image.clone(), &mut frame_times)?;
    // NB! For a single image, ort times will be misleading,
    // as the first time it's used, it does all kinds of lazy init.
    log::debug!("{frame_times:?}");
    // Save output: image & bboxes.
    let img_output_path = path.with_extension("out.jpg");
    img.save(img_output_path)?;
    let bbox_output_path = path.with_extension("out.json");
    let frame_meta = FrameMeta {
        pts: 0,
        dts: 0,
        bboxes_by_class: bboxes,
    };
    serde_json::to_writer(std::fs::File::create(bbox_output_path)?, &frame_meta)?;

    Ok(())
}

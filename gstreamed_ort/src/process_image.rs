use std::path::Path;

use gstreamed_common::frame_times::FrameTimes;
use ort::Session;

use crate::inference;

/// Performs inference on a single image file.
pub fn process_image(path: &Path, session: &Session) -> anyhow::Result<()> {
    let mut frame_times = FrameTimes::default();

    // Read image.
    let og_image = image::open(path)?;

    // Process image.
    let img = inference::infer_on_image(session, None, og_image.clone(), &mut frame_times)?;
    // NB! For a single image, ort times will be misleading,
    // as the first time it's used, it does all kinds of lazy init.
    log::debug!("{frame_times:?}");
    // Save output.
    let output_path = path.with_extension("out.jpg");
    img.save(output_path)?;

    Ok(())
}

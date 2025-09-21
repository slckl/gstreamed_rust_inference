use std::time::Instant;

use fast_image_resize::{ResizeOptions, Resizer};
use image::{DynamicImage, GenericImageView, RgbImage};
use inference_common::tracker::{similari::prelude::Sort, unflatten_bboxes};
use inference_common::{
    annotate::annotate_image_with_bboxes,
    bbox::{BBoxesByClass, Bbox},
    coco_classes,
    frame_times::FrameTimes,
    img_dimensions::ImgDimensions,
};
use ndarray::{Array, Array4, CowArray};
use ort::session::Session;
use ort::value::TensorRef;
use ort_common::yolo_parser::parse_predictions;

/// Transforms the input `image` by converting colors, resizing and loading the image buffer into an [Array].
///
/// Returns the scaled image inside ndarray [Array4] and scaled dims inside [ImgDimensions].
fn preprocess_image(
    image: &DynamicImage,
    target_dims: ImgDimensions,
) -> anyhow::Result<(Array4<f32>, ImgDimensions)> {
    log::debug!("image.dimensions: {:?}", image.dimensions());
    log::debug!("image.color: {:?}", image.color());

    // Convert image to rgb8 to ensure pixel values are just that.
    let image = image.to_rgb8();

    // Resize image to our target size.
    // Target size is not the model input size, but based on the smallest ratio between input and target dims.
    let og_dims: ImgDimensions = image.dimensions().into();
    let ratio = (target_dims.width / og_dims.width).min(target_dims.height / og_dims.height);
    log::debug!("scale ratio: {ratio:?}");
    let scaled_dims = og_dims.scale(ratio);

    // Use `fast_image_resize` crate to resize the image.
    // It has unsafe, but it is way faster than plain `image`, unfortunately...
    let mut scaled_image = fast_image_resize::images::Image::new(
        scaled_dims.width as u32,
        scaled_dims.height as u32,
        fast_image_resize::PixelType::U8x3,
    );

    let mut resizer = Resizer::new();

    let image = DynamicImage::ImageRgb8(image);
    resizer.resize(
        &image,
        &mut scaled_image,
        &ResizeOptions::new().resize_alg(fast_image_resize::ResizeAlg::Nearest),
    )?;

    let scaled_image = RgbImage::from_raw(
        scaled_dims.width as u32,
        scaled_dims.height as u32,
        scaled_image.into_vec(),
    )
    .unwrap();

    // FIXME resize with image crate below is way slower than fast image resize above
    // let scaled_image = image::imageops::resize(
    //     &image,
    //     scaled_dims.width as u32,
    //     scaled_dims.height as u32,
    //     image::imageops::FilterType::Nearest,
    // );
    log::debug!("scaled_image.dimensions: {:?}", scaled_image.dimensions());

    // Load it into ndarray.
    // Array shape: [bsz, channels, height, width];
    let target_shape = [
        1,
        3,
        target_dims.height as usize,
        target_dims.width as usize,
    ];
    let mut image_array = Array::zeros(target_shape);

    // Copy over the pixels starting from the top
    for (x, y, rgb) in scaled_image.enumerate_pixels() {
        let x = x as usize;
        let y = y as usize;
        let [r, g, b] = rgb.0;
        image_array[[0, 0, y, x]] = (r as f32) / 255.0;
        image_array[[0, 1, y, x]] = (g as f32) / 255.0;
        image_array[[0, 2, y, x]] = (b as f32) / 255.0;
    }

    Ok((image_array, scaled_dims))
}

pub fn infer_on_image(
    session: &mut Session,
    tracker: Option<&mut Sort>,
    og_image: DynamicImage,
    frame_times: &mut FrameTimes,
) -> anyhow::Result<(DynamicImage, BBoxesByClass)> {
    // FIXME determine target_dims based on model?
    let model_input_dims = ImgDimensions::new(640f32, 384f32);

    let start = Instant::now();
    let (scaled_image_array, scaled_dims) = preprocess_image(&og_image, model_input_dims)?;
    frame_times.buffer_resize = start.elapsed();

    // Load image into ndarray, and that into ort.
    let start = Instant::now();
    let scaled_image_array = CowArray::from(scaled_image_array).into_dyn();
    log::debug!("image_array.shape: {:?}", scaled_image_array.shape());
    log::debug!("image_array.strides: {:?}", scaled_image_array.strides());

    let input = ort::inputs![TensorRef::from_array_view(&scaled_image_array)?];
    frame_times.buffer_to_tensor = start.elapsed();

    // Now, we can finally run inference.
    let start = Instant::now();
    let outputs = session.run(input)?;
    let outputs = outputs[0].try_extract_array::<f32>()?;
    frame_times.forward_pass = start.elapsed();
    // output shape is 1 x 84 x 5040
    // AKA [bsz, embedding, anchors]
    // embedding is 4 bbox "coords" (center_x, center_y, width, height) + 80 COCO classes long
    log::debug!("got outputs: {outputs:?}");

    // Parse and annotate outputs.
    let conf_threshold = 0.25;
    let nms_threshold = 0.45;
    let bboxes = parse_predictions(
        outputs,
        scaled_dims,
        coco_classes::NAMES.len() as u32,
        conf_threshold,
        nms_threshold,
        frame_times,
    )?;
    log::debug!("{bboxes:?}");
    log::debug!(
        "after nms bboxes, len: {:?}",
        bboxes.iter().map(|v| v.len()).sum::<usize>()
    );

    // Perform tracking.
    let mut tracked_bboxes: Option<Vec<Bbox>> = None;
    if let Some(tracker) = tracker {
        let start = Instant::now();
        tracked_bboxes = Some(inference_common::tracker::predict_tracked_bboxes(
            tracker,
            scaled_dims,
            &bboxes,
        ));
        frame_times.tracking = start.elapsed();
    }
    log::debug!("{tracked_bboxes:?}");

    // Annotate the original image and print boxes information.
    let start = Instant::now();
    let legend_size = 14;

    // Map tracked bboxes back to per class bbox vec...
    let bboxes = match tracked_bboxes {
        Some(tracked) => unflatten_bboxes(tracked),
        None => bboxes,
    };

    let annotated = annotate_image_with_bboxes(
        og_image,
        scaled_dims.width as usize,
        scaled_dims.height as usize,
        legend_size,
        &bboxes,
    );
    frame_times.annotation = start.elapsed();

    Ok((annotated, bboxes))
}

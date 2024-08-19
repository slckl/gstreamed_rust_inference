use std::time::Instant;

use fast_image_resize::{ResizeOptions, Resizer};
use gstreamed_common::{frame_times::FrameTimes, img_dimensions::ImgDimensions};
use image::{DynamicImage, GenericImageView, RgbImage};
use ndarray::{Array, Array4, CowArray, Dimension};

use crate::yolo_parser::report_detect;

// FIXME this function does not quite work right, I think...
/// Creates a new image from the given [Array].
/// It is assumed that the array contains a single image.
fn image_from_ndarray<D: Dimension>(
    array: Array<f32, D>,
    width: u32,
    height: u32,
) -> Option<RgbImage> {
    RgbImage::from_vec(
        width,
        height,
        // Unnormalize back to rgb values.
        array
            .into_raw_vec()
            .into_iter()
            .map(|v| (v * 255.0) as u8)
            .collect(),
    )
}

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
    // Init with gray, similar to how ultralytics does it.
    image_array.fill(0.5);
    // Then copy over the pixels starting from the top, leaving the missing parts filled with gray?
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
    session: &ort::Session,
    og_image: DynamicImage,
    frame_times: &mut FrameTimes,
) -> anyhow::Result<DynamicImage> {
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

    let input = ort::inputs![&scaled_image_array]?;
    frame_times.buffer_to_tensor = start.elapsed();

    // Now, we can finally run inference.
    let start = Instant::now();
    let outputs = session.run(input)?;
    let outputs = outputs[0].try_extract_tensor()?;
    frame_times.forward_pass = start.elapsed();
    // output shape is 1 x 84 x 5040
    // AKA [bsz, embedding, anchors]
    // embedding is 4 bbox "coords" (center_x, center_y, width, height) + 80 COCO classes long
    log::debug!("got outputs: {outputs:?}");

    // parse and annotate outputs
    let conf_threshold = 0.25;
    let img = report_detect(
        outputs,
        og_image,
        scaled_dims,
        conf_threshold,
        0.45,
        14,
        frame_times,
    )?;

    Ok(img)
}

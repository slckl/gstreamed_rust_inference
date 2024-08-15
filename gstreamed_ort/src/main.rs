mod yolo_parser;

use std::ops::Deref;

use crate::yolo_parser::report_detect;
use image::imageops::{resize, FilterType};
use image::{DynamicImage, GenericImageView, RgbImage};
use ndarray::{Array, Array4, ArrayView, CowArray, Dimension, IxDyn};
use ort::tensor::OrtOwnedTensor;
use ort::{ExecutionProvider, GraphOptimizationLevel, OrtResult, SessionBuilder, Value};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

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

    let scaled_image = resize(
        &image,
        scaled_dims.width as u32,
        scaled_dims.height as u32,
        FilterType::Triangle,
    );
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

    let ort_env = ort::Environment::builder()
        .with_name("yolov8")
        .with_execution_providers([ExecutionProvider::CPU(Default::default())])
        .build()?
        .into_arc();

    let session = SessionBuilder::new(&ort_env)?
        // .with_optimization_level(GraphOptimizationLevel::Level1)?
        // .with_intra_threads(1)?
        .with_model_from_file("_models/yolov8s.640x360.cpu.onnx")?;
    log::debug!("session: {session:?}");

    // FIXME determine target_dims based on model?
    let model_input_dims = ImgDimensions::new(640f32, 384f32);
    // read image
    let og_image = image::open("sample.jpg").unwrap();

    let (scaled_image_array, scaled_dims) = preprocess_image(&og_image, model_input_dims)?;
    log::debug!("image_array: {scaled_image_array:?}");

    // check that image still makes sense
    let test_image = image_from_ndarray(
        scaled_image_array.clone(),
        scaled_dims.width as u32,
        scaled_dims.height as u32,
    )
    .unwrap();
    test_image.save("test.jpg").unwrap();

    let image_array = CowArray::from(scaled_image_array).into_dyn();
    log::debug!("image_array: {image_array:?}");
    log::debug!("image_array.shape: {:?}", image_array.shape());
    log::debug!("image_array.strides: {:?}", image_array.strides());
    // read into ndarray
    let input = vec![Value::from_array(session.allocator(), &image_array)?];

    // and run
    let outputs: Vec<Value> = session.run(input)?;
    let outputs: OrtOwnedTensor<f32, _> = outputs[0].try_extract()?;
    let outputs = outputs.view();
    let outputs: &ArrayView<f32, IxDyn> = outputs.deref();
    // output shape is 1 x 84 x 5040
    // AKA [bsz, embedding, anchors]
    // embedding is 4 bbox "coords" (center_x, center_y, width, height) + 80 COCO classes long
    log::debug!("got outputs: {outputs:?}");

    // parse and annotate outputs
    let conf_threshold = 0.25;
    let img = report_detect(
        outputs.into(),
        og_image,
        scaled_dims,
        conf_threshold,
        0.45,
        14,
    )
    .unwrap();
    img.save("output.jpg")?;

    // TODO warmup with synthetic image of the same dims

    Ok(())
}

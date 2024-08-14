mod yolo_parser;

use crate::yolo_parser::report_detect;
use image::imageops::{resize, FilterType};
use image::{DynamicImage, GenericImageView, RgbImage};
use ndarray::{Array, Array4, CowArray, Dimension};
use ort::tensor::OrtOwnedTensor;
use ort::{ExecutionProvider, GraphOptimizationLevel, OrtResult, SessionBuilder, Value};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

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
        array.into_raw_vec().into_iter().map(|v| v as u8).collect(),
    )
}

/// Transforms the input `image` by converting colors, resizing and loading the image buffer into an [Array].
fn preprocess_image(
    image: &DynamicImage,
    target_image_dims: (u32, u32),
) -> anyhow::Result<Array4<f32>> {
    log::debug!("image.dimensions: {:?}", image.dimensions());
    log::debug!("image.color: {:?}", image.color());

    // Convert image to rgb8 to ensure pixel values are just that.
    let image = image.to_rgb8();

    // Resize image to our target size.
    let scaled_image = resize(
        &image,
        target_image_dims.0,
        target_image_dims.1,
        FilterType::Triangle,
    );

    // Load it into ndarray.
    // FIXME we should do better here - loading raw vec is, uhh, not so optimal
    // FIXME don't hardcode shape here
    let image_array = Array4::from_shape_vec(
        // (1, 640, 384, 3),
        (1, 3, 384, 640),
        scaled_image
            .into_raw()
            .into_iter()
            // Normalization: cast from u8 to f32 and convert color from [0; 255] to [0.0; 1.0]
            .map(|v| (v as f32) * (1.0 / 255.0))
            .collect(),
    )?;

    Ok(image_array)
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
    let target_dims = (640, 384);
    // read image
    let image = image::open("sample.jpg").unwrap();

    let image_array = preprocess_image(&image, target_dims)?;
    // dbg!(&image_array);
    // if true {
    //     std::process::exit(0);
    // }

    // check that image still makes sense
    let test_image = image_from_ndarray(image_array.clone(), target_dims.0, target_dims.1).unwrap();
    test_image.save("test.jpg").unwrap();

    let image_array = CowArray::from(image_array).into_dyn();
    log::debug!("image_array.shape: {:?}", image_array.shape());
    log::debug!("image_array.strides: {:?}", image_array.strides());
    // read into ndarray
    // let input: &CowArray = wrapped.into();
    let input = vec![Value::from_array(session.allocator(), &image_array)?];

    // and run
    let outputs: Vec<Value> = session.run(input)?;
    let outputs: OrtOwnedTensor<f32, _> = outputs[0].try_extract()?;
    let output_view = outputs.view();
    // output shape is 1 x 84 x 5040
    log::debug!("got outputs: {outputs:?}");
    // parse and annotate outputs
    let img = report_detect(output_view.view(), image, 640, 384, 0.25, 0.45, 14).unwrap();
    img.save("output.jpg")?;

    // TODO warmup with synthetic image of the same dims

    Ok(())
}

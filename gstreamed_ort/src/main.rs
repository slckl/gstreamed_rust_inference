mod yolo_parser;

use crate::yolo_parser::report_detect;
use image::imageops::{resize, FilterType};
use image::{DynamicImage, GenericImageView, RgbImage};
use ndarray::{Array, Array4, CowArray, Dimension};
use ort::tensor::OrtOwnedTensor;
use ort::{ExecutionProvider, GraphOptimizationLevel, OrtResult, SessionBuilder, Value};

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

fn main() -> anyhow::Result<()> {
    let ort_env = ort::Environment::builder()
        .with_name("yolov8")
        .with_execution_providers([ExecutionProvider::CPU(Default::default())])
        .build()?
        .into_arc();

    let session = SessionBuilder::new(&ort_env)?
        .with_optimization_level(GraphOptimizationLevel::Level1)?
        .with_intra_threads(1)?
        .with_model_from_file("_models/yolov8s.640x360.cpu.onnx")?;
    println!("session: {session:?}");

    // read image
    let image = image::open("sample.jpg").unwrap();

    println!("image.dimensions: {:?}", image.dimensions());
    println!("image.color: {:?}", image.color());
    let image = image.into_rgb8();

    // resize image to fit what we got
    // our model is 640 x 384
    let scaled_image = resize(&image, 640, 384, FilterType::CatmullRom);

    // copy image into ndarray
    // TODO do we need to normalize to 0..1, when converting u8 to f32?
    // and permute axes to match inputs of our model
    let image_array = Array4::from_shape_vec(
        (1, 640, 384, 3),
        scaled_image
            .into_raw()
            .into_iter()
            // cast from u8 to f32 and convert color from [0; 255] to [0.0; 1.0]
            .map(|v| (v as f32) * (1.0 / 255.0))
            .collect(),
    )
    .unwrap()
    .permuted_axes([0, 3, 2, 1]);

    // check that image still makes sense
    let test_image = image_from_ndarray(image_array.clone(), 640, 384).unwrap();
    test_image.save("test.jpg").unwrap();

    let image_array = CowArray::from(image_array).into_dyn();
    println!("image_array.shape: {:?}", image_array.shape());
    // read into ndarray
    // let input: &CowArray = wrapped.into();
    let input = vec![Value::from_array(session.allocator(), &image_array)?];

    // and run
    let outputs: Vec<Value> = session.run(input)?;
    let outputs: OrtOwnedTensor<f32, _> = outputs[0].try_extract()?;
    let output_view = outputs.view();
    // output shape is 1 x 84 x 5040
    println!("got outputs: {outputs:?}");
    // parse and annotate outputs
    let img = report_detect(
        output_view.view(),
        DynamicImage::ImageRgb8(image),
        640,
        384,
        0.25,
        0.45,
        14,
    )
    .unwrap();
    img.save("output.jpg")?;

    // TODO warmup with synthetic image of the same dims

    Ok(())
}

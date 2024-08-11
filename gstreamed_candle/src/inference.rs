//! Roughly corresponds to the logic required for `report_detect` function
//! in yolov8 example code in candle repo.

use std::path::PathBuf;
use std::time::Instant;

use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::object_detection::{non_maximum_suppression, Bbox, KeyPoint};
use clap::ValueEnum;
use gstreamed_common::annotate::annotate_image_with_bboxes;
use image::DynamicImage;

use crate::{
    frame_times::FrameTimes,
    yolov8::{Multiples, YoloV8},
};

// TODO move this to args
#[derive(Clone, Copy, ValueEnum, Debug)]
pub enum Which {
    N,
    S,
    M,
    L,
    X,
}

fn model(which: Which) -> anyhow::Result<PathBuf> {
    // download model from hf hub, cache it locally
    let api = hf_hub::api::sync::Api::new()?;
    let api = api.model("lmz/candle-yolo-v8".into());
    let size = match which {
        Which::N => "n",
        Which::S => "s",
        Which::M => "m",
        Which::L => "l",
        Which::X => "x",
    };
    let path = api.get(&format!("yolov8{size}.safetensors"))?;
    Ok(path)
}

pub fn load_model(which: Which, device: &Device) -> anyhow::Result<YoloV8> {
    let multiples = match which {
        Which::N => Multiples::n(),
        Which::S => Multiples::s(),
        Which::M => Multiples::m(),
        Which::L => Multiples::l(),
        Which::X => Multiples::x(),
    };
    let model = model(which)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model], DType::F32, device)? };
    // let weights = unsafe { candle_core::safetensors::MmapedFile::new(model)? };
    // let weights = weights.deserialize()?;
    // let vb = VarBuilder::from_safetensors(vec![weights], DType::F32, &Device::Cpu);
    let model = YoloV8::load(vb, multiples, 80)?;
    Ok(model)
}

fn report_detect(
    pred: &Tensor,
    img: DynamicImage,
    w: usize,
    h: usize,
    confidence_threshold: f32,
    nms_threshold: f32,
    legend_size: u32,
    frame_times: &mut FrameTimes,
) -> anyhow::Result<DynamicImage> {
    // println!("initial pred.shape: {:?}", pred.shape());
    let start = Instant::now();
    let (pred_size, npreds) = pred.dims2()?;
    let nclasses = pred_size - 4;
    // The bounding boxes grouped by (maximum) class index.
    let mut bboxes: Vec<Vec<Bbox<Vec<KeyPoint>>>> = (0..nclasses).map(|_| vec![]).collect();
    // Extract the bounding boxes for which confidence is above the threshold.
    // Since we compute bboxes on cpu, transfer whole prediction tensor to cpu, so it's not done inside a loop.
    let pred = pred.to_device(&Device::Cpu)?;
    for index in 0..npreds {
        let pred = pred.i((.., index))?;
        // println!("pred.shape: {:?}", pred.shape());
        let pred = Vec::<f32>::try_from(pred)?;
        // println!("pred.len(): {}", pred.len());
        // std::io::stdout().flush().unwrap();
        let confidence = *pred[4..].iter().max_by(|x, y| x.total_cmp(y)).unwrap();
        if confidence > confidence_threshold {
            let mut class_index = 0;
            for i in 0..nclasses {
                if pred[4 + i] > pred[4 + class_index] {
                    class_index = i
                }
            }
            if pred[class_index + 4] > 0. {
                let bbox = Bbox {
                    xmin: pred[0] - pred[2] / 2.,
                    ymin: pred[1] - pred[3] / 2.,
                    xmax: pred[0] + pred[2] / 2.,
                    ymax: pred[1] + pred[3] / 2.,
                    confidence,
                    data: vec![],
                };
                bboxes[class_index].push(bbox)
            }
        }
    }
    frame_times.bbox_extraction = start.elapsed();

    let start = Instant::now();
    non_maximum_suppression(&mut bboxes, nms_threshold);
    frame_times.nms = start.elapsed();

    // Annotate the original image and print boxes information.
    let start = Instant::now();
    let annotated = annotate_image_with_bboxes(img, w, h, legend_size, &bboxes);
    frame_times.annotation = start.elapsed();

    Ok(annotated)
}

/// Run yolov8 inference, and draw detections on top of the frame.
///
/// Largely copypasta of report_detect in candle yolov8 example code.
pub fn process_frame(
    frame: DynamicImage,
    model: &YoloV8,
    device: &Device,
    conf_thresh: f32,
    nms_thresh: f32,
    legend_size: u32,
    frame_times: &mut FrameTimes,
) -> anyhow::Result<DynamicImage> {
    // Resize buffer to match input size of model.
    let start = Instant::now();
    let (width, height) = {
        let w = frame.width() as usize;
        let h = frame.height() as usize;
        if w < h {
            let w = w * 640 / h;
            // Sizes have to be divisible by 32.
            (w / 32 * 32, 640)
        } else {
            let h = h * 640 / w;
            (640, h / 32 * 32)
        }
    };
    let img = frame.resize_exact(
        width as u32,
        height as u32,
        image::imageops::FilterType::Nearest,
        // image::imageops::FilterType::CatmullRom,
    );
    frame_times.buffer_resize = start.elapsed();

    // Convert image buffer to tensor.
    let start = Instant::now();
    let data = img.into_rgb8().into_raw();
    let image_t = Tensor::from_vec(data, (height, width, 3), device)?.permute((2, 0, 1))?;
    let image_t = (image_t.unsqueeze(0)?.to_dtype(DType::F32)? * (1. / 255.))?;
    frame_times.buffer_to_tensor = start.elapsed();

    // Run forward pass.
    let start = Instant::now();
    let predictions = model.forward(&image_t)?.squeeze(0)?;
    frame_times.forward_pass = start.elapsed();

    // Process predictions and draw overlays.
    let image_t = report_detect(
        &predictions,
        frame,
        width,
        height,
        conf_thresh,
        nms_thresh,
        legend_size,
        frame_times,
    )?;

    // Return processed image tensor.
    Ok(image_t)
}

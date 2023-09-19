//! Roughly corresponds to the logic required for `report_detect` function
//! in yolov8 example code in candle repo.

use std::path::PathBuf;

use crate::coco_classes;
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::object_detection::{non_maximum_suppression, Bbox, KeyPoint};
use clap::ValueEnum;
use image::DynamicImage;

use crate::yolov8::{Multiples, YoloV8};

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

pub fn load_model(which: Which) -> anyhow::Result<YoloV8> {
    let multiples = match which {
        Which::N => Multiples::n(),
        Which::S => Multiples::s(),
        Which::M => Multiples::m(),
        Which::L => Multiples::l(),
        Which::X => Multiples::x(),
    };
    let model = model(which)?;
    let weights = unsafe { candle_core::safetensors::MmapedFile::new(model)? };
    let weights = weights.deserialize()?;
    let vb = VarBuilder::from_safetensors(vec![weights], DType::F32, &Device::Cpu);
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
) -> anyhow::Result<DynamicImage> {
    let (pred_size, npreds) = pred.dims2()?;
    let nclasses = pred_size - 4;
    // The bounding boxes grouped by (maximum) class index.
    let mut bboxes: Vec<Vec<Bbox<Vec<KeyPoint>>>> = (0..nclasses).map(|_| vec![]).collect();
    // Extract the bounding boxes for which confidence is above the threshold.
    for index in 0..npreds {
        let pred = Vec::<f32>::try_from(pred.i((.., index))?)?;
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

    non_maximum_suppression(&mut bboxes, nms_threshold);

    // Annotate the original image and print boxes information.
    let (initial_h, initial_w) = (img.height(), img.width());
    let w_ratio = initial_w as f32 / w as f32;
    let h_ratio = initial_h as f32 / h as f32;
    let mut img = img.to_rgb8();
    let font = Vec::from(include_bytes!("roboto-mono-stripped.ttf") as &[u8]);
    let font = rusttype::Font::try_from_vec(font);
    for (class_index, bboxes_for_class) in bboxes.iter().enumerate() {
        for b in bboxes_for_class.iter() {
            println!("{}: {:?}", coco_classes::NAMES[class_index], b);
            let xmin = (b.xmin * w_ratio) as i32;
            let ymin = (b.ymin * h_ratio) as i32;
            let dx = (b.xmax - b.xmin) * w_ratio;
            let dy = (b.ymax - b.ymin) * h_ratio;
            if dx >= 0. && dy >= 0. {
                imageproc::drawing::draw_hollow_rect_mut(
                    &mut img,
                    imageproc::rect::Rect::at(xmin, ymin).of_size(dx as u32, dy as u32),
                    image::Rgb([255, 0, 0]),
                );
            }
            if legend_size > 0 {
                if let Some(font) = font.as_ref() {
                    imageproc::drawing::draw_filled_rect_mut(
                        &mut img,
                        imageproc::rect::Rect::at(xmin, ymin).of_size(dx as u32, legend_size),
                        image::Rgb([170, 0, 0]),
                    );
                    let legend = format!(
                        "{}   {:.0}%",
                        coco_classes::NAMES[class_index],
                        100. * b.confidence
                    );
                    imageproc::drawing::draw_text_mut(
                        &mut img,
                        image::Rgb([255, 255, 255]),
                        xmin,
                        ymin,
                        rusttype::Scale::uniform(legend_size as f32 - 1.),
                        font,
                        &legend,
                    )
                }
            }
        }
    }
    Ok(DynamicImage::ImageRgb8(img))
}

/// Run yolov8 inference, and draw detections on top of the frame.
///
/// Largely copypasta of report_detect in candle yolov8 example code.
pub fn process_frame(
    frame: DynamicImage,
    model: &YoloV8,
    conf_thresh: f32,
    nms_thresh: f32,
    legend_size: u32,
) -> anyhow::Result<DynamicImage> {
    // inference
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
    let image_t = {
        let img = frame.resize_exact(
            width as u32,
            height as u32,
            image::imageops::FilterType::CatmullRom,
        );
        let data = img.to_rgb8().into_raw();
        Tensor::from_vec(
            data,
            (img.height() as usize, img.width() as usize, 3),
            &Device::Cpu,
        )?
        .permute((2, 0, 1))?
    };
    let image_t = (image_t.unsqueeze(0)?.to_dtype(DType::F32)? * (1. / 255.))?;
    let predictions = model.forward(&image_t)?.squeeze(0)?;
    // process predictions and draw overlays
    let image_t = report_detect(
        &predictions,
        frame,
        width,
        height,
        conf_thresh,
        nms_thresh,
        legend_size,
    )?;
    // return processed frame
    Ok(image_t)
}

use gstreamed_common::{
    annotate::annotate_image_with_bboxes,
    bbox::{non_maximum_suppression, Bbox, KeyPoint},
};
use image::DynamicImage;
use ndarray::{s, ArrayView, IxDyn};

fn parse_predictions(
    preds: ArrayView<f32, IxDyn>,
    conf_threshold: f32,
    nms_threshold: f32,
) -> Vec<Vec<Bbox<Vec<KeyPoint>>>> {
    // in candle case we have:
    // initial pred.shape: [84, 4620]
    // pred.shape: [84]
    // pred.len(): 84
    let shape = preds.shape();
    println!("preds.shape: {shape:?}");
    // TODO study wtf they doing in ultralytics yolo rust example
    let (pred_size, npreds) = (shape[1], shape[2]);
    let nclasses = pred_size - 4;
    // The bounding boxes grouped by (maximum) class index.
    let mut bboxes: Vec<Vec<Bbox<Vec<KeyPoint>>>> = (0..nclasses).map(|_| vec![]).collect();
    // Extract the bounding boxes for which confidence is above the threshold.
    for index in 0..npreds {
        let pred_view = preds.slice(s![.., .., index]);
        println!("pred_view.shape: {:?}", pred_view.shape());
        let pred: Vec<f32> = pred_view.iter().copied().collect();
        // println!("pred: {pred:?}");
        // std::process::exit(0);
        let confidence = *pred[4..].iter().max_by(|x, y| x.total_cmp(y)).unwrap();
        if confidence > conf_threshold {
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
                if bbox.xmin < 0.0 || bbox.ymin < 0.0 || bbox.xmax < 0.0 || bbox.ymax < 0.0 {
                    eprintln!("bbox with negative coords: {bbox:?}");
                    eprintln!("from preds: {pred:?}");
                } else {
                    bboxes[class_index].push(bbox)
                }
            }
        }
    }
    // std::process::exit(0);
    // nms
    non_maximum_suppression(&mut bboxes, nms_threshold);
    bboxes
}

/// Equivalent of report_detect for candle, but using ndarray ArrayView as input
/// so this can be used for ort predictions.
pub fn report_detect(
    pred_view: ArrayView<f32, IxDyn>,
    img: DynamicImage,
    w: usize,
    h: usize,
    confidence_threshold: f32,
    nms_threshold: f32,
    legend_size: u32,
) -> anyhow::Result<DynamicImage> {
    let bboxes = parse_predictions(pred_view, confidence_threshold, nms_threshold);

    // Annotate the original image and print boxes information.
    Ok(annotate_image_with_bboxes(img, w, h, legend_size, &bboxes))
}

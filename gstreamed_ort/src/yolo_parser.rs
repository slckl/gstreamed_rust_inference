use std::time::Instant;

use gstreamed_common::{
    annotate::annotate_image_with_bboxes,
    bbox::{non_maximum_suppression, Bbox},
    coco_classes,
    frame_times::FrameTimes,
};
use image::DynamicImage;
use ndarray::{s, ArrayView, Axis, Dim, IxDyn};

use crate::ImgDimensions;

/// Parse yolov8 predictions via `ort`.
pub fn parse_predictions(
    preds: ArrayView<f32, IxDyn>,
    scaled_dims: ImgDimensions,
    num_clases: u32,
    conf_threshold: f32,
    nms_threshold: f32,
    frame_times: &mut FrameTimes,
) -> anyhow::Result<Vec<Vec<Bbox>>> {
    // preds.shape: [bsz, embedding, anchors]
    // [1, 84, 5040]
    // TODO batch support with another loop outside

    let start = Instant::now();
    log::debug!("preds.shape: {:?}", preds.shape());
    // Get rid of the first axis.
    // Need to specify full dimensions here so rust can infer slices correctly later.
    let preds: ArrayView<f32, Dim<[usize; 2]>> = preds.slice(s![0, .., ..]);
    // Gives us a shape of [84, 5040].
    log::debug!("preds2.shape: {:?}", preds.shape());

    let mut bboxes_per_class: Vec<Vec<Bbox>> = vec![Vec::new(); num_clases as usize];
    for pred in preds.axis_iter(Axis(1)) {
        log::trace!("pred.shape: {:?}", pred.shape());
        // Separate bbox and class values.
        // First 4 values correspond to bbox cx, cy, w, h
        const BBOX_OFFSET: usize = 4;
        let bbox = pred.slice(s![0..BBOX_OFFSET]);
        let clss = pred.slice(s![BBOX_OFFSET..BBOX_OFFSET + num_clases as usize]);

        // Determine top1 class and its confidence.
        let mut max_class_id = 0;
        let mut max_confidence = 0f32;
        for (idx, cls_conf) in clss.into_iter().enumerate() {
            if cls_conf > &max_confidence {
                max_confidence = *cls_conf;
                max_class_id = idx;
            }
        }

        log::trace!("max class id {max_class_id:?}: {max_confidence:?}");

        // Check confidence > threshold.
        if max_confidence < conf_threshold {
            continue;
        }

        let cx = bbox[0];
        let cy = bbox[1];
        let w = bbox[2];
        let h = bbox[3];

        let xmin = cx - w / 2.;
        let ymin = cy - h / 2.;
        let xmax = xmin + w;
        let ymax = ymin + h;

        // Bound coords to scaled dimensions, so bboxes don't go outside the image.
        let y_bbox = Bbox {
            xmin: xmin.max(0.0f32).min(scaled_dims.width),
            ymin: ymin.max(0.0f32).min(scaled_dims.height),
            xmax: xmax.max(0.0f32).min(scaled_dims.width),
            ymax: ymax.max(0.0f32).min(scaled_dims.height),
            confidence: max_confidence,
            data: vec![],
        };

        bboxes_per_class[max_class_id].push(y_bbox);
    }
    frame_times.bbox_extraction = start.elapsed();

    // nms
    let start = Instant::now();
    log::debug!(
        "be4 nms bboxes, len: {:?}",
        bboxes_per_class.iter().map(|v| v.len()).sum::<usize>()
    );
    non_maximum_suppression(&mut bboxes_per_class, nms_threshold);
    frame_times.nms = start.elapsed();

    Ok(bboxes_per_class)
}

/// Equivalent of report_detect for candle, but using ndarray ArrayView as input
/// so this can be used for ort predictions.
pub fn report_detect(
    predicted: ArrayView<f32, IxDyn>,
    img: DynamicImage,
    scaled_dims: ImgDimensions,
    confidence_threshold: f32,
    nms_threshold: f32,
    legend_size: u32,
    frame_times: &mut FrameTimes,
) -> anyhow::Result<DynamicImage> {
    let bboxes = parse_predictions(
        predicted,
        scaled_dims,
        coco_classes::NAMES.len() as u32,
        confidence_threshold,
        nms_threshold,
        frame_times,
    )?;
    log::debug!("{bboxes:?}");
    log::debug!(
        "after nms bboxes, len: {:?}",
        bboxes.iter().map(|v| v.len()).sum::<usize>()
    );

    let start = Instant::now();
    // Annotate the original image and print boxes information.
    let annotated = annotate_image_with_bboxes(
        img,
        scaled_dims.width as usize,
        scaled_dims.height as usize,
        legend_size,
        &bboxes,
    );
    frame_times.annotation = start.elapsed();

    Ok(annotated)
}

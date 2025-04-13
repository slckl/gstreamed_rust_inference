use std::time::Instant;

use inference_common::{
    bbox::{Bbox, non_maximum_suppression},
    frame_times::FrameTimes,
    img_dimensions::ImgDimensions,
};
use ndarray::{ArrayView, Axis, Dim, IxDyn, s};

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
            detector_confidence: max_confidence,
            tracker_confidence: 0f32,
            data: vec![],
            class: max_class_id,
            tracker_id: None,
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

//! Common tracker interface, params etc.

use std::sync::Mutex;

use crate::bbox::{BBoxesByClass, Bbox};
use crate::coco_classes;
use crate::img_dimensions::ImgDimensions;
use similari::prelude::PositionalMetricType::IoU;
use similari::prelude::{SortTrack, Universal2DBox};
use similari::trackers::sort::metric::DEFAULT_MINIMAL_SORT_CONFIDENCE;
use similari::{prelude::Sort, trackers::sort::DEFAULT_SORT_IOU_THRESHOLD};

// `similari` re-export so types can be named etc.
pub use similari;

/// Creates a SORT tracker with some default params, largely untuned.
pub fn sort_tracker() -> Mutex<Sort> {
    Mutex::new(Sort::new(
        1,
        1,
        10,
        IoU(DEFAULT_SORT_IOU_THRESHOLD),
        DEFAULT_MINIMAL_SORT_CONFIDENCE,
        None,
        1.0 / 20.0,
        1.0 / 160.0,
    ))
}

/// Maps from [SortTrack] back to our [Bbox].
pub fn tracks_to_bboxes(tracks: &[SortTrack], scaled_dims: ImgDimensions) -> Vec<Bbox> {
    let mut out = Vec::with_capacity(tracks.len());
    for track in tracks {
        let tracked_bbox = &track.predicted_bbox;
        // We use similari custom object id support for class ids.
        let class_id = track.custom_object_id.unwrap();
        let id = track.id;

        // Map from similary bbox to our bbox.
        let cx = tracked_bbox.xc;
        let cy = tracked_bbox.yc;
        let h = tracked_bbox.height;
        let aspect = tracked_bbox.aspect;
        let w = aspect * h;

        let xmin = cx - w / 2f32;
        let ymin = cy - h / 2f32;
        let xmax = xmin + w;
        let ymax = ymin + h;

        out.push(Bbox {
            xmin: xmin.max(0.0f32).min(scaled_dims.width),
            ymin: ymin.max(0.0f32).min(scaled_dims.height),
            xmax: xmax.max(0.0f32).min(scaled_dims.width),
            ymax: ymax.max(0.0f32).min(scaled_dims.height),
            // FIXME this, unfortunately, does not retain og yolo confidence...
            detector_confidence: track.observed_bbox.confidence,
            // FIXME tracker confidence is always very high?
            //  does it not work for sort?
            tracker_confidence: tracked_bbox.confidence,
            data: vec![],
            class: class_id as usize,
            tracker_id: Some(id as i64),
        });
    }
    out
}

/// Predicts [SortTrack]s using the given [Sort] tracker and
/// observed `bboxes_per_class`.
pub fn predict_tracks(tracker: &mut Sort, bboxes_per_class: &[Vec<Bbox>]) -> Vec<SortTrack> {
    let bboxes_4_tracking: Vec<_> = bboxes_per_class
        .iter()
        .flatten()
        .map(|bbox| {
            (
                Universal2DBox::ltwh(
                    bbox.xmin,
                    bbox.ymin,
                    bbox.xmax - bbox.xmin,
                    bbox.ymax - bbox.ymin,
                ),
                Some(bbox.class as i64),
            )
        })
        .collect();
    tracker.predict(&bboxes_4_tracking)
}

/// Predicts tracked [Bbox]es from the input `bboxes_per_class`,
/// using the given `tracker` and `scaled_dims`.
pub fn predict_tracked_bboxes(
    tracker: &mut Sort,
    scaled_dims: ImgDimensions,
    bboxes_per_class: &[Vec<Bbox>],
) -> Vec<Bbox> {
    let tracks = predict_tracks(tracker, bboxes_per_class);
    log::trace!("{tracks:?}");
    tracks_to_bboxes(&tracks, scaled_dims)
}

/// Transform a flat list of [Bbox] back into bboxes grouped by class.
///
/// NB! Currently hardcoded to use coco classes.
pub fn unflatten_bboxes(flat_bboxes: Vec<Bbox>) -> BBoxesByClass {
    let mut bboxes_by_class = vec![Vec::new(); coco_classes::NAMES.len()];
    for tracked_bbox in flat_bboxes {
        bboxes_by_class[tracked_bbox.class].push(tracked_bbox);
    }
    bboxes_by_class
}

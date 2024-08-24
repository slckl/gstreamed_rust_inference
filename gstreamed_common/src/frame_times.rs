use std::fmt::{self, Debug};
use std::iter::Sum;
use std::time::Duration;

/// Various timings for processing a single video frame.
#[derive(Default, Clone, PartialEq)]
pub struct FrameTimes {
    pub frame_to_buffer: Duration,
    pub buffer_resize: Duration,
    pub buffer_to_tensor: Duration,
    pub forward_pass: Duration,
    pub bbox_extraction: Duration,
    pub nms: Duration,
    pub tracking: Duration,
    pub annotation: Duration,
    pub buffer_to_frame: Duration,
}

impl FrameTimes {
    pub fn total(&self) -> Duration {
        self.frame_to_buffer
            + self.buffer_resize
            + self.buffer_to_tensor
            + self.forward_pass
            + self.bbox_extraction
            + self.nms
            + self.tracking
            + self.annotation
            + self.buffer_to_frame
    }

    pub fn uniform(ms: u64) -> Self {
        FrameTimes {
            frame_to_buffer: Duration::from_millis(ms),
            buffer_resize: Duration::from_millis(ms),
            buffer_to_tensor: Duration::from_millis(ms),
            forward_pass: Duration::from_millis(ms),
            bbox_extraction: Duration::from_millis(ms),
            nms: Duration::from_millis(ms),
            tracking: Duration::from_millis(ms),
            annotation: Duration::from_millis(ms),
            buffer_to_frame: Duration::from_millis(ms),
        }
    }
}

impl Debug for FrameTimes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "
            total: {:?},
            frame_to_buffer: {:?},
            buffer_resize: {:?},
            buffer_to_tensor: {:?},
            forward_pass: {:?},
            bbox_extraction: {:?},
            nms: {:?},
            tracking: {:?},
            annotation: {:?},
            buffer_to_frame: {:?}
            ",
            self.total(),
            self.frame_to_buffer,
            self.buffer_resize,
            self.buffer_to_tensor,
            self.forward_pass,
            self.bbox_extraction,
            self.nms,
            self.tracking,
            self.annotation,
            self.buffer_to_frame,
        )
    }
}

impl Sum for FrameTimes {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(FrameTimes::default(), |mut acc, x| {
            acc.frame_to_buffer += x.frame_to_buffer;
            acc.buffer_resize += x.buffer_resize;
            acc.buffer_to_tensor += x.buffer_to_tensor;
            acc.forward_pass += x.forward_pass;
            acc.bbox_extraction += x.bbox_extraction;
            acc.nms += x.nms;
            acc.tracking += x.tracking;
            acc.annotation += x.annotation;
            acc.buffer_to_frame += x.buffer_to_frame;
            acc
        })
    }
}

/// Basic frame times collector for aggregated stats.
#[derive(Default)]
pub struct AggregatedTimes(Vec<FrameTimes>);

impl AggregatedTimes {
    pub fn push(&mut self, frame_times: FrameTimes) {
        self.0.push(frame_times);
    }

    /// Computes
    pub fn avg(&self, ignore_first: bool) -> FrameTimes {
        let sum: FrameTimes = self
            .0
            .iter()
            .skip(if ignore_first { 1 } else { 0 })
            .cloned()
            .sum();

        let mut n = self.0.len();
        if ignore_first {
            n = n.saturating_sub(1);
        }
        let n = n as u32;

        if n == 0 {
            return FrameTimes::default();
        }

        FrameTimes {
            frame_to_buffer: sum.frame_to_buffer / n,
            buffer_resize: sum.buffer_resize / n,
            buffer_to_tensor: sum.buffer_to_tensor / n,
            forward_pass: sum.forward_pass / n,
            bbox_extraction: sum.bbox_extraction / n,
            nms: sum.nms / n,
            tracking: sum.tracking / n,
            annotation: sum.annotation / n,
            buffer_to_frame: sum.buffer_to_frame / n,
        }
    }

    pub fn min(&self, ignore_first: bool) -> FrameTimes {
        let frame_times_iter = self.0.iter().skip(if ignore_first { 1 } else { 0 });

        let mut empty = true;
        let mut min = FrameTimes::uniform(u64::MAX);
        let comp = Duration::min;

        for ft in frame_times_iter {
            empty = false;

            min.frame_to_buffer = comp(min.frame_to_buffer, ft.frame_to_buffer);
            min.buffer_resize = comp(min.buffer_resize, ft.buffer_resize);
            min.buffer_to_tensor = comp(min.buffer_to_tensor, ft.buffer_to_tensor);
            min.forward_pass = comp(min.forward_pass, ft.forward_pass);
            min.bbox_extraction = comp(min.bbox_extraction, ft.bbox_extraction);
            min.nms = comp(min.nms, ft.nms);
            min.tracking = comp(min.tracking, ft.tracking);
            min.annotation = comp(min.annotation, ft.annotation);
            min.buffer_to_frame = comp(min.buffer_to_frame, ft.buffer_to_frame);
        }

        if !empty {
            min
        } else {
            FrameTimes::default()
        }
    }

    pub fn max(&self, ignore_first: bool) -> FrameTimes {
        let frame_times_iter = self.0.iter().skip(if ignore_first { 1 } else { 0 });

        let mut empty = true;
        let mut max = FrameTimes::uniform(0);
        let comp = Duration::max;

        for ft in frame_times_iter {
            empty = false;

            max.frame_to_buffer = comp(max.frame_to_buffer, ft.frame_to_buffer);
            max.buffer_resize = comp(max.buffer_resize, ft.buffer_resize);
            max.buffer_to_tensor = comp(max.buffer_to_tensor, ft.buffer_to_tensor);
            max.forward_pass = comp(max.forward_pass, ft.forward_pass);
            max.bbox_extraction = comp(max.bbox_extraction, ft.bbox_extraction);
            max.nms = comp(max.nms, ft.nms);
            max.tracking = comp(max.tracking, ft.tracking);
            max.annotation = comp(max.annotation, ft.annotation);
            max.buffer_to_frame = comp(max.buffer_to_frame, ft.buffer_to_frame);
        }

        if !empty {
            max
        } else {
            FrameTimes::default()
        }
    }
}

#[test]
fn aggregate_avg() {
    let f1 = FrameTimes::uniform(4000);
    let f2 = FrameTimes::uniform(8000);

    let mut agg = AggregatedTimes::default();
    agg.push(f1);
    agg.push(f2);

    let avg = agg.avg(false);
    let tgt = FrameTimes::uniform(6000);
    assert_eq!(avg, tgt);
}

#[test]
fn aggregate_avg_ignore_first() {
    let f1 = FrameTimes::uniform(40000);
    let f2 = FrameTimes::uniform(200);

    let mut agg = AggregatedTimes::default();
    agg.push(f1);
    agg.push(f2);

    let avg = agg.avg(true);
    let tgt = FrameTimes::uniform(200);
    assert_eq!(avg, tgt);
}

#[test]
fn aggregate_min_and_max() {
    let mut f1 = FrameTimes::uniform(600);
    f1.forward_pass = Duration::from_millis(5000);
    f1.bbox_extraction = Duration::from_millis(200);
    f1.nms = Duration::from_millis(300);

    let mut f2 = FrameTimes::uniform(900);
    f2.frame_to_buffer = Duration::from_millis(100);
    f2.forward_pass = Duration::from_millis(499);

    let f3 = FrameTimes::uniform(500);

    let mut agg = AggregatedTimes::default();
    agg.push(f1);
    agg.push(f2);
    agg.push(f3);

    let min = agg.min(false);
    let mut min_tgt = FrameTimes::uniform(500);
    min_tgt.frame_to_buffer = Duration::from_millis(100);
    min_tgt.forward_pass = Duration::from_millis(499);
    min_tgt.bbox_extraction = Duration::from_millis(200);
    min_tgt.nms = Duration::from_millis(300);
    assert_eq!(min, min_tgt);

    let max = agg.max(false);
    let mut max_tgt = FrameTimes::uniform(900);
    max_tgt.frame_to_buffer = Duration::from_millis(600);
    max_tgt.forward_pass = Duration::from_millis(5000);
    assert_eq!(max, max_tgt);
}

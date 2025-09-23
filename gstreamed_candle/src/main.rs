mod inference;
mod yolov8;

use crate::inference::Which;
use candle_core::Device;
use clap::Parser;
use gstreamed_common::discovery;
use gstreamed_common::pipeline::build_pipeline;
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer::MessageView;
use inference_common::frame_meta::FrameMeta;
use inference_common::frame_times::{AggregatedTimes, FrameTimes};
use inference_common::img_dimensions::ImgDimensions;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Debug, Parser)]
pub struct Args {
    /// Path to input image (.jpeg/.png) or video file (.mp4/.mkv).
    input: PathBuf,
    #[arg(long, action, default_value = "false")]
    cuda: bool,
    // TODO dtype switch
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Initialize logging.
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    gst::init()?;

    let device = if args.cuda {
        Device::new_cuda(0)?
    } else {
        Device::Cpu
    };

    // Load models using hf-hub.
    let which = Which::S;
    let model = inference::load_model(which, &device)?;

    // Branch on file extension: video vs image.
    let ext = args
        .input
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase());

    match ext.as_deref() {
        Some("mp4") | Some("mkv") => {
            // Video processing path using gstreamer pipeline.
            // First, find out resolution of input file.
            let file_info = discovery::discover(&args.input)?;
            log::info!("{file_info:?}");
            let frame_dims = ImgDimensions::new(file_info.width as f32, file_info.height as f32);

            let agg_times = Arc::new(Mutex::new(AggregatedTimes::default()));

            // Use tracker for candle pipeline, too.
            let tracker = inference_common::tracker::sort_tracker();

            // Build gst pipeline, which performs inference using the loaded model.
            let scoped_agg = Arc::clone(&agg_times);
            let pipeline = build_pipeline(
                args.input.to_str().unwrap(),
                args.input.with_extension("out.mkv").to_str().unwrap(),
                false,
                move |buf| {
                    let mut agg_times = scoped_agg.lock().unwrap();
                    inference::process_buffer(
                        frame_dims,
                        &model,
                        &device,
                        &tracker,
                        &mut agg_times,
                        buf,
                    );
                },
            )?;

            // Make it play and listen to events to know when it's done.
            pipeline.set_state(gst::State::Playing).unwrap();

            let bus = pipeline.bus().unwrap();
            for msg in bus.iter_timed(gst::ClockTime::NONE) {
                match msg.view() {
                    MessageView::Error(err) => {
                        pipeline.debug_to_dot_file(gst::DebugGraphDetails::all(), "pipeline.error");
                        let name = err.src().map(|e| e.name().to_string());
                        log::error!("Error from element {name:?}: {}", err.error());
                        break;
                    }
                    MessageView::Eos(..) => {
                        log::info!("Pipeline reached end of stream.");
                        break;
                    }
                    _ => (),
                }
            }

            pipeline.set_state(gst::State::Null).unwrap();

            // Print perf stats, ignoring first (outlier) frame.
            let agg = agg_times.lock().unwrap();
            let avg = agg.avg(true);
            log::info!("Average frame times: {avg:?}");

            let min = agg.min(true);
            log::info!("Min frame times: {min:?}");

            let max = agg.max(true);
            log::info!("Max frame times: {max:?}");
        }
        Some("jpg") | Some("jpeg") | Some("png") => {
            // Single image processing path.
            let og_image = image::open(&args.input)?;
            let mut frame_times = FrameTimes::default();

            // Tracking on a single image isn't super meaningful, but we keep it for parity.
            let tracker = inference_common::tracker::sort_tracker();
            let (annotated, bboxes) = {
                let mut t = tracker.lock().unwrap();
                inference::process_frame(
                    og_image,
                    &model,
                    &device,
                    &mut t,
                    0.25,
                    0.45,
                    14,
                    &mut frame_times,
                )?
            };

            // Save output annotated image.
            let img_output_path = args.input.with_extension("out.jpg");
            annotated.save(&img_output_path)?;

            // Save bboxes json.
            let frame_meta = FrameMeta {
                pts: 0,
                dts: 0,
                bboxes_by_class: bboxes,
            };
            let bbox_output_path = args.input.with_extension("out.json");
            serde_json::to_writer(std::fs::File::create(bbox_output_path)?, &frame_meta)?;

            log::debug!("Frame times (single image; includes warmup): {frame_times:?}");
        }
        Some(unk) => {
            log::error!("Unhandled file extension: {unk}");
        }
        None => {
            log::error!(
                "Input path does not have valid file extension: {:?}",
                args.input
            );
        }
    }

    Ok(())
}

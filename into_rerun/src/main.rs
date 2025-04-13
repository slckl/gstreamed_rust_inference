use std::{
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
};

use clap::Parser;
use inference_common::video_meta::VideoMeta;
use rerun::{AssetVideo, Boxes2D, VideoFrameReference};

#[derive(Debug, Parser)]
pub struct Args {
    /// Path to input .json file with bboxes.
    input: PathBuf,
    /// Path to output rerun log file, typical extension `.rrd`.
    output: PathBuf,
}

fn read_video_meta(input: &Path) -> VideoMeta {
    let file = File::open(input).unwrap();
    let reader = BufReader::new(file);
    serde_json::from_reader(reader).unwrap()
}

fn main() {
    let args: Args = Args::parse();

    let video_meta = read_video_meta(&args.input);
    // Create a rerun log, stream it to disk at output location.
    let rec = rerun::RecordingStreamBuilder::new("gstreamed_rust_inference")
        .save(&args.output)
        .unwrap();

    // Add input video as a video asset.
    let video_asset = AssetVideo::from_file_path(video_meta.input_file).unwrap();
    // let frame_timestamps = video_asset.read_frame_timestamps_ns().unwrap();
    rec.log("video", &video_asset).unwrap();

    // Log per frame data.
    for (idx, frame) in video_meta.frames.iter().enumerate() {
        rec.log("video", &VideoFrameReference::new(frame.pts as i64))
            .unwrap();
        // rr.log(format!("bboxes/{idx}"), Boxes2D::)
    }

    println!("Finished writing rerun log to {:?}", args.output);
}

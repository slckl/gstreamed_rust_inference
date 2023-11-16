use crate::discovery::FileInfo;
use crate::inference;
use crate::yolov8::YoloV8;
use candle_core::Device;
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer::{glib, PadProbeData, PadProbeReturn, PadProbeType};
use image::{DynamicImage, RgbImage};
use std::io::Write;
use std::time::Instant;

fn file_src_bin(input_file: &str) -> Result<gst::Element, glib::BoolError> {
    let bin = gst::Bin::new();
    // filesrc -> decodebin -> queue
    // filesrc, well, it reads a file
    let source = gst::ElementFactory::make_with_name("filesrc", None)?;
    source.set_property_from_str("location", input_file);

    // decodebin automagically determines the input format
    // and constructs and links the appropriate decoder
    let decode_bin = gst::ElementFactory::make_with_name("decodebin", None)?;

    // finally, we use a queue so we have a late linking target
    // because decodebin's automagic needs to read the file and so is constructed "late"
    let queue = gst::ElementFactory::make_with_name("queue", None)?;

    bin.add_many([&source, &decode_bin, &queue])?;
    // eager link filesrc -> decodebin
    gst::Element::link_many([&source, &decode_bin])?;

    // construct ghost src pad for the bin we cooking here
    let queue_src = queue.static_pad("src").unwrap();
    let bin_ghost_src_pad = gst::GhostPad::with_target(&queue_src)?;

    bin.add_pad(&bin_ghost_src_pad)?;

    // perform late linking by adding a callback to decodebin's signal for "pad-added" event
    // create a glib weak ref to queue, so we can safely look it up inside callback.
    let queue_weak = queue.downgrade();
    decode_bin.connect_pad_added(move |_decode_bin, pad| {
        // check if queue's still around, it should be
        if let Some(queue) = queue_weak.upgrade() {
            let sink_pad = queue
                .compatible_pad(pad, None)
                .expect("Compatible sink pad not found for late linking");
            pad.link(&sink_pad)
                .expect("Could not link decodebin src pad to queue sink pad");
        } else {
            eprintln!("Late linking: file_src_bin queue element has been dropped");
        }
    });

    Ok(bin.upcast())
}

// filesrc -> decodebin -> [candle] -> queue -> encode -> mkvmux
pub fn build_pipeline(
    input_file: &str,
    file_info: FileInfo,
    model: YoloV8,
    device: Device,
) -> Result<gst::Pipeline, glib::BoolError> {
    let pipeline = gst::Pipeline::new();

    // filesrc -> caps_filter -> video_convert -> [candle] -> queue -> encode -> mkvmux
    let file_src_bin = file_src_bin(input_file)?;
    // add video_convert -> caps filter to force RGB buffers
    let video_convert = gst::ElementFactory::make_with_name("videoconvert", None)?;
    let caps = gst::caps::Caps::builder(glib::gstr!("video/x-raw"))
        .field("format", "RGB")
        .build();
    let caps_filter = gst::ElementFactory::make_with_name("capsfilter", None)?;
    caps_filter.set_property("caps", &caps);

    let queue = gst::ElementFactory::make_with_name("queue", None)?;
    // perform inference between file_src_bin and queue using a probe on queue src pad
    let queue_src = queue.static_pad("src").unwrap();
    println!("queue_src caps: {:?}", queue_src.caps());
    queue_src.add_probe(PadProbeType::BUFFER, move |_pad, pad_probe_info| {
        // we're interested in the buffer
        if let Some(PadProbeData::Buffer(buffer)) = &mut pad_probe_info.data {
            let start = Instant::now();
            // read buffer into an image
            let image = {
                let readable = buffer.map_readable().unwrap();
                let readable_vec = readable.to_vec();

                // buffer size is: width x height x 3
                let image = RgbImage::from_vec(
                    file_info.width as u32,
                    file_info.height as u32,
                    readable_vec,
                )
                .unwrap();
                // debug code
                // image.save("./output.jpg").unwrap();
                // std::process::exit(0);
                DynamicImage::ImageRgb8(image)
            };
            println!(
                "Read buffer into vector in {:.4} ms",
                start.elapsed().as_secs_f32() * 1000.0
            );

            // process it using some model + draw overlays on the output image
            let start = Instant::now();
            let processed =
                inference::process_frame(image, &model, &device, 0.25, 0.45, 14).unwrap();
            println!(
                "Processed frame in {:.4} ms",
                start.elapsed().as_secs_f32() * 1000.0
            );
            // processed.save("./output.jpg").unwrap();
            // std::process::exit(0);

            // overwrite the buffer with our overlaid processed image
            let start = Instant::now();
            let buffer_mut = buffer.get_mut().unwrap();
            let mut writable = buffer_mut.map_writable().unwrap();
            let mut dst = writable.as_mut_slice();
            dst.write_all(processed.to_rgb8().as_raw()).unwrap();
            println!(
                "Wrote processed frame to buffer in {:.4} ms",
                start.elapsed().as_secs_f32() * 1000.0
            );
        }

        PadProbeReturn::Ok
    });

    // during dev, just dump output to autodisplaysink
    let display_convert = gst::ElementFactory::make_with_name("videoconvert", None)?;
    let display_sink = gst::ElementFactory::make_with_name("autovideosink", None)?;
    pipeline.add_many([
        &file_src_bin,
        &video_convert,
        &caps_filter,
        &queue,
        &display_convert,
        &display_sink,
    ])?;
    gst::Element::link_many([
        &file_src_bin,
        &video_convert,
        &caps_filter,
        &queue,
        &display_convert,
        &display_sink,
    ])?;

    // TODO video output to file
    // // cpu encoder
    // let encoder = gst::ElementFactory::make_with_name("x264enc", None)?;
    // // mkv muxer
    // let mkv_mux = gst::ElementFactory::make_with_name("matroskamux", None)?;
    // let file_sink = gst::ElementFactory::make_with_name("filesink", None)?;

    Ok(pipeline)
}

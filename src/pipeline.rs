use crate::yolov8::YoloV8;
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer::{glib, PadProbeData, PadProbeReturn, PadProbeType};

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
pub fn build_pipeline(input_file: &str, model: YoloV8) -> Result<gst::Pipeline, glib::BoolError> {
    let pipeline = gst::Pipeline::new();

    // filesrc -> [candle] -> queue -> encode -> mkvmux
    let file_src_bin = file_src_bin(input_file)?;
    let queue = gst::ElementFactory::make_with_name("queue", None)?;
    // perform inference between file_src_bin and queue using a probe on queue src pad
    let queue_src = queue.static_pad("src").unwrap();
    queue_src.add_probe(PadProbeType::BUFFER, move |_pad, pad_probe_info| {
        // we're interested in the buffer
        if let Some(PadProbeData::Buffer(buffer)) = &pad_probe_info.data {
            // TODO read this into a DynamicImage
            // TODO call inference::process_frame
            // TODO overwrite the buffer with the processed frame
            println!("TODO actually do yolo stuff");
        }

        PadProbeReturn::Ok
    });

    // during dev, just dump output to autodisplaysink
    let display_sink = gst::ElementFactory::make_with_name("autovideosink", None)?;
    pipeline.add_many([&file_src_bin, &queue, &display_sink])?;
    gst::Element::link_many([&file_src_bin, &queue, &display_sink])?;

    // TODO video output to file
    // // cpu encoder
    // let encoder = gst::ElementFactory::make_with_name("x264enc", None)?;
    // // mkv muxer
    // let mkv_mux = gst::ElementFactory::make_with_name("matroskamux", None)?;
    // let file_sink = gst::ElementFactory::make_with_name("filesink", None)?;

    Ok(pipeline)
}

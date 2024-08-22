use gstreamer::prelude::*;
use gstreamer::{self as gst, Buffer};
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
/// Builds gst pipeline that takes input video, decodes it, runs inference
/// on the decoded frames, and then annotates the frame with inference output.
///
/// The annotated output is saved into a separate file by default that follows the naming of the `input_file`, but appends `.out.mkv` to filename.
///
/// If `live_playback` is enabled, then we create a parallel branch
/// with a gst `autovideosink`, which usually manages to create a window
/// with live playback of the annotated output.
pub fn build_pipeline(
    input_file: &str,
    live_playback: bool,
    buffer_processor: impl Fn(&mut Buffer) + Send + Sync + 'static,
) -> Result<gst::Pipeline, glib::BoolError> {
    let pipeline = gst::Pipeline::new();

    // filesrc -> caps_filter -> video_convert -> [candle] -> queue -> encode -> mkvmux
    let file_src_bin = file_src_bin(input_file)?;
    // add video_convert -> caps filter to force RGB buffers
    // NB! If we use cuda device, use nvidia magic videoconvert at least once in pipeline
    // so we can handle laptop scenarios (with built-in graphics + cuda).
    let converter_factory = if let Some(factory) = gst::ElementFactory::find("nvvideoconvert") {
        factory
    } else {
        gst::ElementFactory::find("videoconvert").unwrap()
    };
    let video_convert = converter_factory.create().build()?;

    let caps = gst::caps::Caps::builder(glib::gstr!("video/x-raw"))
        .field("format", "RGB")
        .build();
    let caps_filter = gst::ElementFactory::make_with_name("capsfilter", None)?;
    caps_filter.set_property("caps", &caps);

    let queue = gst::ElementFactory::make_with_name("queue", None)?;
    // perform inference between file_src_bin and queue using a probe on queue src pad
    let queue_src = queue.static_pad("src").unwrap();
    // println!("queue_src caps: {:?}", queue_src.caps());
    queue_src.add_probe(PadProbeType::BUFFER, move |_pad, pad_probe_info| {
        // we're interested in the buffer
        if let Some(PadProbeData::Buffer(buffer)) = &mut pad_probe_info.data {
            buffer_processor(buffer);
        }

        PadProbeReturn::Ok
    });

    let encoder_convert = gst::ElementFactory::make_with_name("videoconvert", None)?;
    // let encoder_factory =
    // // if let Some(factory) =
    // // gst::ElementFactory::find("nvh264enc") {
    // //     factory
    // // } else {
    //     gst::ElementFactory::find("x264enc").unwrap();
    // // };
    // let encoder = encoder_factory.create().build()?;
    let encoder = gst::ElementFactory::make_with_name("x264enc", None)?;
    // Default is 2048, which for dynamic videos will look like ass.
    encoder.set_property_from_str("bitrate", "8192");
    let mkv_mux = gst::ElementFactory::make_with_name("matroskamux", None)?;
    let file_sink = gst::ElementFactory::make_with_name("filesink", None)?;
    let output_path = format!("{input_file}.out.mkv");
    file_sink.set_property_from_str("location", &output_path);

    if live_playback {
        let tee = gst::ElementFactory::make_with_name("tee", None)?;
        let display_queue = gst::ElementFactory::make_with_name("queue", Some("display_queue"))?;
        // Make display_queue leaky, so it doesn't block large pipelines.
        display_queue.set_property_from_str("leaky", "downstream");
        let encoder_queue = gst::ElementFactory::make_with_name("queue", Some("encoder_queue"))?;
        let display_convert = gst::ElementFactory::make_with_name("videoconvert", None)?;
        let display_sink = gst::ElementFactory::make_with_name("autovideosink", None)?;

        // Add and link up to tee
        let elements_to_tee = [&file_src_bin, &video_convert, &caps_filter, &queue, &tee];
        pipeline.add_many(elements_to_tee)?;
        gst::Element::link_many(elements_to_tee)?;

        // Add and wire up the 2 output branches.
        // Encoder/output branch.
        let encoder_elements = [
            &encoder_queue,
            &encoder_convert,
            &encoder,
            &mkv_mux,
            &file_sink,
        ];
        pipeline.add_many(encoder_elements)?;
        // Tee -> encoder_queue
        tee.link(&encoder_queue)?;
        // encoder_queue -> ...
        gst::Element::link_many(encoder_elements)?;

        // Live display branch.
        let display_elements = [&display_queue, &display_convert, &display_sink];
        pipeline.add_many(display_elements)?;
        tee.link(&display_queue)?;
        gst::Element::link_many(display_elements)?;
    } else {
        // No live playback, so just wire everything through encoded output.
        let elements = [
            &file_src_bin,
            &video_convert,
            &caps_filter,
            &queue,
            &encoder_convert,
            &encoder,
            &mkv_mux,
            &file_sink,
        ];
        pipeline.add_many(elements)?;
        gst::Element::link_many(elements)?;
    }

    Ok(pipeline)
}

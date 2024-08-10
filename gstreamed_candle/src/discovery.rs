use gstreamer as gst;
use gstreamer::prelude::Displayable;
use gstreamer_pbutils::prelude::DiscovererStreamInfoExt;
use gstreamer_pbutils::{Discoverer, DiscovererInfo, DiscovererStreamInfo};
use std::path::Path;

/// Info about the media in the given file.
#[derive(Debug)]
pub struct FileInfo {
    pub width: i32,
    pub height: i32,
}

fn print_tags(info: &DiscovererInfo) {
    let tags = info.tags();
    match tags {
        Some(taglist) => {
            println!("  {taglist}"); // FIXME use an iterator
        }
        None => {
            println!("  no tags");
        }
    }
}

fn print_stream_info(stream: &DiscovererStreamInfo) {
    println!("Stream: ");

    if let Some(stream_id) = stream.stream_id() {
        println!("  Stream id: {}", stream_id);
    }

    let caps_str = match stream.caps() {
        Some(caps) => caps.to_string(),
        None => String::from("--"),
    };
    println!("  Format: {caps_str}");
}

fn print_discoverer_info(info: &DiscovererInfo) -> anyhow::Result<()> {
    println!("-------");
    println!("URI: {}", info.uri());
    println!("Duration: {}", info.duration().display());
    print_tags(info);
    print_stream_info(&info.stream_info().unwrap());

    let children = info.stream_list();
    println!("Children streams:");
    for child in children {
        print_stream_info(&child);
    }
    println!("-------");

    Ok(())
}

fn discover_resolution_from_stream_info(stream_info: &DiscovererStreamInfo) -> Option<FileInfo> {
    let mut width = None;
    let mut height = None;
    if let Some(caps) = stream_info.caps() {
        let caps_ref = caps.as_ref();
        for struct_ref in caps_ref.iter() {
            for (name, value) in struct_ref.iter() {
                if name == "width" {
                    width = Some(value.get().unwrap());
                }
                if name == "height" {
                    height = Some(value.get().unwrap());
                }
                if let (Some(width), Some(height)) = (width, height) {
                    return Some(FileInfo { width, height });
                }
                // println!("{name:?}: {value:?}");
            }
        }
    }

    None
}

fn discover_resolution(info: &DiscovererInfo) -> anyhow::Result<FileInfo> {
    if let Some(stream_info) = info.stream_info() {
        if let Some(file_info) = discover_resolution_from_stream_info(&stream_info) {
            return Ok(file_info);
        }
    }
    for child_stream in info.stream_list() {
        if let Some(file_info) = discover_resolution_from_stream_info(&child_stream) {
            return Ok(file_info);
        }
    }
    Err(anyhow::anyhow!(
        "No stream with a width/height feature pair discovered"
    ))
}

fn raw_discoverer_info(path: &Path) -> anyhow::Result<DiscovererInfo> {
    let timeout = gst::ClockTime::from_seconds(10);
    let discoverer = Discoverer::new(timeout)?;
    // we need to pass absolute path to discoverer as file uri
    let file_uri = format!("file://{}", path.canonicalize()?.to_str().unwrap());
    Ok(discoverer.discover_uri(&file_uri)?)
}

pub fn discover(path: &Path) -> anyhow::Result<FileInfo> {
    let info = raw_discoverer_info(path)?;
    // print_discoverer_info is str8 copypasta from https://gitlab.freedesktop.org/gstreamer/gstreamer-rs/-/blob/main/examples/src/bin/discoverer.rs
    // useful for debugging, but not necessary
    print_discoverer_info(&info)?;

    // extract stuff we actually care about, which is just the resolution p much
    discover_resolution(&info)
}

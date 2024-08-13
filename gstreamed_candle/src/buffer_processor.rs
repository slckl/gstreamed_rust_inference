use std::io::Write;
use std::time::Instant;

use candle_core::Device;
use gstreamed_common::{discovery::FileInfo, frame_times::FrameTimes};
use gstreamer as gst;
use image::{DynamicImage, RgbImage};

use crate::{inference, pipeline::BufferProcessor, yolov8::YoloV8};

pub struct CandleBufferProcessor {
    pub file_info: FileInfo,
    pub model: YoloV8,
    pub device: Device,
}

impl BufferProcessor for CandleBufferProcessor {
    fn process(&self, buffer: &mut gst::Buffer) {
        let file_info = &self.file_info;
        let model = &self.model;
        let device = &self.device;

        let mut frame_times = FrameTimes::default();

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
        frame_times.frame_to_buffer = start.elapsed();

        // process it using some model + draw overlays on the output image
        let processed =
            inference::process_frame(image, &model, &device, 0.25, 0.45, 14, &mut frame_times)
                .unwrap();

        // processed.save("./output.jpg").unwrap();
        // std::process::exit(0);

        // overwrite the buffer with our overlaid processed image
        let start = Instant::now();
        let buffer_mut = buffer.get_mut().unwrap();
        let mut writable = buffer_mut.map_writable().unwrap();
        let mut dst = writable.as_mut_slice();
        dst.write_all(processed.to_rgb8().as_raw()).unwrap();
        frame_times.buffer_to_frame = start.elapsed();

        log::debug!("{frame_times:?}");
    }
}

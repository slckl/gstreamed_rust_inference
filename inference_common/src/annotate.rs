//! Largely modified candle code.

use crate::{bbox::Bbox, coco_classes};
use image::DynamicImage;

/// Draws bboxes on the given image.
/// Returns the same image (just annotated now).
pub fn annotate_image_with_bboxes(
    og_img: DynamicImage,
    scaled_width: usize,
    scaled_height: usize,
    legend_size: u32,
    bboxes: &[Vec<Bbox>],
) -> DynamicImage {
    let (initial_h, initial_w) = (og_img.height(), og_img.width());
    let w_ratio = initial_w as f32 / scaled_width as f32;
    let h_ratio = initial_h as f32 / scaled_height as f32;
    let font = Vec::from(include_bytes!("roboto-mono-stripped.ttf") as &[u8]);
    let font = ab_glyph::FontRef::try_from_slice(&font);
    let mut img = og_img.into_rgb8();
    for (class_index, bboxes_for_class) in bboxes.iter().enumerate() {
        for b in bboxes_for_class.iter() {
            log::trace!("{}: {:?}", coco_classes::NAMES[class_index], b);
            let xmin = (b.xmin * w_ratio) as i32;
            let ymin = (b.ymin * h_ratio) as i32;
            let dx = (b.xmax - b.xmin) * w_ratio;
            let dy = (b.ymax - b.ymin) * h_ratio;
            if dx >= 0. && dy >= 0. {
                imageproc::drawing::draw_hollow_rect_mut(
                    &mut img,
                    imageproc::rect::Rect::at(xmin, ymin).of_size(dx as u32, dy as u32),
                    image::Rgb([255, 0, 0]),
                );
            }
            if legend_size > 0 {
                if let Ok(font) = font.as_ref() {
                    imageproc::drawing::draw_filled_rect_mut(
                        &mut img,
                        imageproc::rect::Rect::at(xmin, ymin).of_size(dx as u32, legend_size),
                        image::Rgb([170, 0, 0]),
                    );
                    let legend = format!(
                        "{} {:?}   {:.0}% {:.0}%",
                        coco_classes::NAMES[class_index],
                        b.tracker_id,
                        100. * b.detector_confidence,
                        100. * b.tracker_confidence,
                    );
                    imageproc::drawing::draw_text_mut(
                        &mut img,
                        image::Rgb([255, 255, 255]),
                        xmin,
                        ymin,
                        ab_glyph::PxScale::from(legend_size as f32 - 1.),
                        font,
                        &legend,
                    )
                }
            }
        }
    }
    DynamicImage::ImageRgb8(img)
}

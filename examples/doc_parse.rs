//! FusionOcr 高级 API 示例。
//! 运行：cargo run --example doc_parse --release

use ab_glyph::{FontArc, PxScale};
use fusion_ocr::{FusionOcr, FusionOcrConfig, FusionOcrModelPaths, FusionOcrParagraph};
use image::{Rgb, RgbImage};
use imageproc::{
    drawing::{draw_filled_rect_mut, draw_hollow_rect_mut, draw_text_mut, text_size},
    rect::Rect,
};
use std::{error::Error, fs, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let total_start = Instant::now();
    let image = image::open("assets/Page_1_docsmall.com.jpg")?.to_rgb8();
    fs::create_dir_all("outputs")?;

    let paths = FusionOcrModelPaths::new("models/PP-DocLayout_plus-L.onnx")
        .with_text(
            "models/PP-OCRv6_tiny_det.onnx",
            "models/PP-OCRv6_tiny_rec_compact.onnx",
            "models/ppocrv6_tiny_dict.txt",
        )
        .with_formula(
            "models/PP-FormulaNet_plus-S.onnx",
            "models/unimernet_tokens.txt",
        );
    let config = FusionOcrConfig::basic();

    let init_start = Instant::now();
    let mut ocr = FusionOcr::new(paths, config)?;
    let init_elapsed = init_start.elapsed();

    let recognize_start = Instant::now();
    let results = ocr.recognize(&image)?;
    let recognize_elapsed = recognize_start.elapsed();

    let markdown = results
        .iter()
        .filter(|result| !result.content.is_empty())
        .map(|result| result.content.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");
    fs::write("outputs/doc_parse_result.md", &markdown)?;

    let font = FontArc::try_from_slice(include_bytes!("../fonts/DejaVuSans-Bold.ttf"))?;
    let mut output = image.clone();
    for result in &results {
        draw_result(&mut output, &font, result);
    }
    output.save("outputs/doc_parse_result.jpg")?;

    println!("recognized {} configured regions", results.len());
    println!("saved outputs/doc_parse_result.jpg and outputs/doc_parse_result.md");
    println!("\n--- Markdown ---\n{markdown}");
    let timing = ocr.last_timing();
    println!(
        "[timing] init: {:.3?}, recognize: {:.3?}, total: {:.3?}",
        init_elapsed,
        recognize_elapsed,
        total_start.elapsed()
    );
    println!(
        "[timing] layout: {:.3?}, det: {:.3?}, rec: {:.3?}",
        timing.layout, timing.detect, timing.recognize
    );
    Ok(())
}

fn draw_result(out: &mut RgbImage, font: &FontArc, result: &FusionOcrParagraph) {
    let [x1, y1, x2, y2] = result.bbox;
    let (x1, y1, x2, y2) = (x1 as i32, y1 as i32, x2 as i32, y2 as i32);
    let (width, height) = ((x2 - x1) as u32, (y2 - y1) as u32);
    let color = color_for(&result.paragraph_type);
    draw_hollow_rect_mut(out, Rect::at(x1, y1).of_size(width, height), color);
    if width > 2 && height > 2 {
        draw_hollow_rect_mut(
            out,
            Rect::at(x1 + 1, y1 + 1).of_size(width - 2, height - 2),
            color,
        );
    }

    let scale = PxScale::from(18.0);
    let (text_width, text_height) = text_size(scale, font, &result.paragraph_type);
    let tag_x = x1.min((out.width() as i32 - text_width as i32 - 8).max(0));
    let tag_y = (y1 - text_height as i32 - 6).max(0);
    draw_filled_rect_mut(
        out,
        Rect::at(tag_x, tag_y).of_size(text_width + 8, text_height + 4),
        color,
    );
    draw_text_mut(
        out,
        Rgb([0, 0, 0]),
        tag_x + 4,
        tag_y + 2,
        scale,
        font,
        &result.paragraph_type,
    );
}

fn color_for(paragraph_type: &str) -> Rgb<u8> {
    match paragraph_type {
        "paragraph_title" => Rgb([220, 20, 60]),
        "text" => Rgb([30, 30, 200]),
        "formula" => Rgb([0, 150, 0]),
        _ => Rgb([70, 130, 180]),
    }
}

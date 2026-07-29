//! 文档版面区域检测，基于 PP-DocLayout_plus-L（RT-DETR 架构）。
//! 预处理/类别标签与 `models/PP-DocLayout_plus-L.yml` 保持一致。

use image::{RgbImage, imageops::FilterType};
use ndarray::{Array2, Array4};
use ort::{inputs, session::Session, session::builder::GraphOptimizationLevel, value::Tensor};
use std::{error::Error, path::Path};

/// 类别标签，顺序与 PP-DocLayout_plus-L.yml 中 `label_list` 一致。
pub const LABELS: [&str; 20] = [
    "paragraph_title",
    "image",
    "text",
    "number",
    "abstract",
    "content",
    "figure_title",
    "formula",
    "table",
    "reference",
    "doc_title",
    "footnote",
    "header",
    "algorithm",
    "footer",
    "seal",
    "chart",
    "formula_number",
    "aside_text",
    "reference_content",
];

/// 模型输入边长（yml: target_size [800, 800], keep_ratio: false）。
const INPUT_SIZE: usize = 800;

/// 一个检测到的版面区域。
#[derive(Debug, Clone)]
pub struct LayoutBox {
    /// 类别索引，对应 [`LABELS`]。
    pub label_id: usize,
    /// 类别名称。
    pub label: &'static str,
    /// 置信度。
    pub score: f32,
    /// `[x1, y1, x2, y2]`，原图像素坐标，已裁剪到图像范围内。
    pub bbox: [f32; 4],
}

/// 版面检测器。
pub struct LayoutPredictor {
    session: Session,
    /// 置信度阈值（yml: draw_threshold）。
    threshold: f32,
}

impl LayoutPredictor {
    /// 从 ONNX 模型文件创建检测器，置信度阈值默认 0.5。
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self, Box<dyn Error>> {
        Self::with_threshold(model_path, 0.5)
    }

    /// 从 ONNX 模型文件创建检测器，并指定置信度阈值。
    pub fn with_threshold<P: AsRef<Path>>(
        model_path: P,
        threshold: f32,
    ) -> Result<Self, Box<dyn Error>> {
        // layout 与 det 并行运行（见 pipeline），各占一半逻辑核，避免过度订阅。
        let session = crate::ocr::session_builder((crate::ocr::available_threads() / 2).max(1))?
            .with_optimization_level(GraphOptimizationLevel::All)?
            .commit_from_file(model_path)?;
        Ok(Self { session, threshold })
    }

    /// 对整页图像做版面检测，返回按置信度过滤后的区域列表。
    pub fn predict(&mut self, img: &RgbImage) -> Result<Vec<LayoutBox>, Box<dyn Error>> {
        let (orig_w, orig_h) = (img.width() as f32, img.height() as f32);

        // Preprocess: Resize(800x800) -> NormalizeImage(mean=0, std=1 即 /255) -> Permute(CHW)，BGR 通道序
        let resized = image::imageops::resize(
            img,
            INPUT_SIZE as u32,
            INPUT_SIZE as u32,
            FilterType::Triangle,
        );
        let mut image = Array4::<f32>::zeros((1, 3, INPUT_SIZE, INPUT_SIZE));
        let plane = INPUT_SIZE * INPUT_SIZE;
        let buf = image.as_slice_mut().expect("Array4::zeros 是连续内存");
        for (i, px) in resized.as_raw().chunks_exact(3).enumerate() {
            buf[i] = px[2] as f32 / 255.0; // B
            buf[plane + i] = px[1] as f32 / 255.0; // G
            buf[2 * plane + i] = px[0] as f32 / 255.0; // R
        }
        let im_shape =
            Array2::<f32>::from_shape_vec((1, 2), vec![INPUT_SIZE as f32, INPUT_SIZE as f32])?;
        let scale_factor = Array2::<f32>::from_shape_vec(
            (1, 2),
            vec![INPUT_SIZE as f32 / orig_h, INPUT_SIZE as f32 / orig_w],
        )?;

        let outputs = self.session.run(inputs![
            "image" => Tensor::from_array(image)?,
            "im_shape" => Tensor::from_array(im_shape)?,
            "scale_factor" => Tensor::from_array(scale_factor)?,
        ])?;

        // fetch_name_0: [N, 6] = [label_id, score, x1, y1, x2, y2]（原图坐标）
        // fetch_name_1: [batch] 有效框数量
        let (shape, boxes) = outputs["fetch_name_0"].try_extract_tensor::<f32>()?;
        let num_boxes = match outputs["fetch_name_1"].try_extract_tensor::<i32>() {
            Ok((_, nums)) if !nums.is_empty() => (nums[0] as usize).min(shape[0] as usize),
            _ => shape[0] as usize,
        };

        let mut results = Vec::new();
        for row in boxes[..num_boxes * 6].chunks_exact(6) {
            let (label_id, score) = (row[0] as usize, row[1]);
            if score < self.threshold || label_id >= LABELS.len() {
                continue;
            }
            let x1 = row[2].clamp(0.0, orig_w);
            let y1 = row[3].clamp(0.0, orig_h);
            let x2 = row[4].clamp(0.0, orig_w);
            let y2 = row[5].clamp(0.0, orig_h);
            if x2 <= x1 || y2 <= y1 {
                continue;
            }
            results.push(LayoutBox {
                label_id,
                label: LABELS[label_id],
                score,
                bbox: [x1, y1, x2, y2],
            });
        }
        Ok(deduplicate_overlapping_boxes(results))
    }
}

/// RT-DETR 可能为同一个区域输出多个不同类别。对于位置几乎相同的候选框，
/// 只保留置信度最高的一个；普通的相交或嵌套区域不受影响。
fn deduplicate_overlapping_boxes(mut boxes: Vec<LayoutBox>) -> Vec<LayoutBox> {
    const DUPLICATE_IOU_THRESHOLD: f32 = 0.9;

    boxes.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<LayoutBox> = Vec::with_capacity(boxes.len());
    for candidate in boxes {
        if kept.iter().any(|existing| {
            intersection_over_union(candidate.bbox, existing.bbox) >= DUPLICATE_IOU_THRESHOLD
        }) {
            continue;
        }
        kept.push(candidate);
    }
    kept
}

fn intersection_over_union(a: [f32; 4], b: [f32; 4]) -> f32 {
    let intersection_width = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let intersection_height = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let intersection = intersection_width * intersection_height;
    if intersection <= 0.0 {
        return 0.0;
    }

    let area_a = (a[2] - a[0]) * (a[3] - a[1]);
    let area_b = (b[2] - b[0]) * (b[3] - b[1]);
    intersection / (area_a + area_b - intersection)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout_box(label_id: usize, score: f32, bbox: [f32; 4]) -> LayoutBox {
        LayoutBox {
            label_id,
            label: LABELS[label_id],
            score,
            bbox,
        }
    }

    #[test]
    fn overlapping_candidates_keep_only_highest_confidence() {
        let boxes = vec![
            layout_box(10, 0.67, [100.0, 100.0, 500.0, 160.0]),
            layout_box(12, 0.66, [100.0, 100.0, 500.0, 160.0]),
            layout_box(2, 0.60, [102.0, 101.0, 498.0, 159.0]),
        ];

        let kept = deduplicate_overlapping_boxes(boxes);

        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].label, "doc_title");
        assert_eq!(kept[0].score, 0.67);
    }

    #[test]
    fn distinct_and_partially_overlapping_regions_are_preserved() {
        let boxes = vec![
            layout_box(2, 0.90, [0.0, 0.0, 100.0, 100.0]),
            layout_box(4, 0.80, [50.0, 50.0, 150.0, 150.0]),
            layout_box(0, 0.70, [0.0, 120.0, 100.0, 160.0]),
        ];

        let kept = deduplicate_overlapping_boxes(boxes);

        assert_eq!(kept.len(), 3);
    }
}

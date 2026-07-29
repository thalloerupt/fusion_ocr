//! Fusion OCR 高级流水线 API。

use crate::ocr::{
    formula::recognition::FormulaRecognizer,
    layout::predictor::{LABELS, LayoutBox, LayoutPredictor},
    text::{
        detection::{TextDetector, TextLine},
        recognition::TextRecognizer,
    },
};
use image::RgbImage;
use std::{
    error::Error,
    fmt,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

pub type FusionOcrResult<T> = Result<T, Box<dyn Error>>;

/// FusionOcr 所需模型与字典路径。
#[derive(Debug, Clone)]
pub struct FusionOcrModelPaths {
    pub layout_model: PathBuf,
    pub text_det_model: Option<PathBuf>,
    pub text_rec_model: Option<PathBuf>,
    pub text_rec_dict: Option<PathBuf>,
    pub formula_model: Option<PathBuf>,
    pub formula_tokens: Option<PathBuf>,
}

impl FusionOcrModelPaths {
    pub fn new(layout_model: impl Into<PathBuf>) -> Self {
        Self {
            layout_model: layout_model.into(),
            text_det_model: None,
            text_rec_model: None,
            text_rec_dict: None,
            formula_model: None,
            formula_tokens: None,
        }
    }

    pub fn with_text(
        mut self,
        det_model: impl Into<PathBuf>,
        rec_model: impl Into<PathBuf>,
        rec_dict: impl Into<PathBuf>,
    ) -> Self {
        self.text_det_model = Some(det_model.into());
        self.text_rec_model = Some(rec_model.into());
        self.text_rec_dict = Some(rec_dict.into());
        self
    }

    pub fn with_formula(
        mut self,
        formula_model: impl Into<PathBuf>,
        formula_tokens: impl Into<PathBuf>,
    ) -> Self {
        self.formula_model = Some(formula_model.into());
        self.formula_tokens = Some(formula_tokens.into());
        self
    }
}

/// DocLayout 类别处理清单。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutClassConfig {
    pub paragraph_title: bool,
    pub image: bool,
    pub text: bool,
    pub number: bool,
    pub r#abstract: bool,
    pub content: bool,
    pub figure_title: bool,
    pub formula: bool,
    pub table: bool,
    pub reference: bool,
    pub doc_title: bool,
    pub footnote: bool,
    pub header: bool,
    pub algorithm: bool,
    pub footer: bool,
    pub seal: bool,
    pub chart: bool,
    pub formula_number: bool,
    pub aside_text: bool,
    pub reference_content: bool,
}

impl LayoutClassConfig {
    /// 基础预设：仅处理 paragraph_title 与 text。
    pub fn basic() -> Self {
        Self {
            paragraph_title: true,
            text: true,
            doc_title: true,
            r#abstract: true,
            figure_title: true,
            ..Self::none()
        }
    }

    pub fn none() -> Self {
        Self {
            paragraph_title: false,
            image: false,
            text: false,
            number: false,
            r#abstract: false,
            content: false,
            figure_title: false,
            formula: false,
            table: false,
            reference: false,
            doc_title: false,
            footnote: false,
            header: false,
            algorithm: false,
            footer: false,
            seal: false,
            chart: false,
            formula_number: false,
            aside_text: false,
            reference_content: false,
        }
    }

    pub fn all() -> Self {
        Self {
            paragraph_title: true,
            image: true,
            text: true,
            number: true,
            r#abstract: true,
            content: true,
            figure_title: true,
            formula: true,
            table: true,
            reference: true,
            doc_title: true,
            footnote: true,
            header: true,
            algorithm: true,
            footer: true,
            seal: true,
            chart: true,
            formula_number: true,
            aside_text: true,
            reference_content: true,
        }
    }

    pub fn is_enabled(&self, label: &str) -> bool {
        match label {
            "paragraph_title" => self.paragraph_title,
            "image" => self.image,
            "text" => self.text,
            "number" => self.number,
            "abstract" => self.r#abstract,
            "content" => self.content,
            "figure_title" => self.figure_title,
            "formula" => self.formula,
            "table" => self.table,
            "reference" => self.reference,
            "doc_title" => self.doc_title,
            "footnote" => self.footnote,
            "header" => self.header,
            "algorithm" => self.algorithm,
            "footer" => self.footer,
            "seal" => self.seal,
            "chart" => self.chart,
            "formula_number" => self.formula_number,
            "aside_text" => self.aside_text,
            "reference_content" => self.reference_content,
            _ => false,
        }
    }

    /// 按 DocLayout 标签名启用或关闭类别。未知标签返回 false。
    pub fn set_enabled(&mut self, label: &str, enabled: bool) -> bool {
        let target = match label {
            "paragraph_title" => &mut self.paragraph_title,
            "image" => &mut self.image,
            "text" => &mut self.text,
            "number" => &mut self.number,
            "abstract" => &mut self.r#abstract,
            "content" => &mut self.content,
            "figure_title" => &mut self.figure_title,
            "formula" => &mut self.formula,
            "table" => &mut self.table,
            "reference" => &mut self.reference,
            "doc_title" => &mut self.doc_title,
            "footnote" => &mut self.footnote,
            "header" => &mut self.header,
            "algorithm" => &mut self.algorithm,
            "footer" => &mut self.footer,
            "seal" => &mut self.seal,
            "chart" => &mut self.chart,
            "formula_number" => &mut self.formula_number,
            "aside_text" => &mut self.aside_text,
            "reference_content" => &mut self.reference_content,
            _ => return false,
        };
        *target = enabled;
        true
    }

    pub fn with_enabled(mut self, label: &str, enabled: bool) -> Self {
        self.set_enabled(label, enabled);
        self
    }

    fn needs_rec_model(&self) -> bool {
        LABELS
            .iter()
            .any(|label| self.is_enabled(label) && is_text_label(label))
    }

    fn needs_det_model(&self) -> bool {
        LABELS
            .iter()
            .any(|label| self.is_enabled(label) && is_multiline_text_label(label))
    }
}

impl Default for LayoutClassConfig {
    fn default() -> Self {
        Self::basic()
    }
}

/// Fusion OCR 高级配置。
#[derive(Debug, Clone)]
pub struct FusionOcrConfig {
    pub classes: LayoutClassConfig,
    pub layout_confidence_threshold: f32,
    pub rec_confidence_threshold: f32,
    pub rec_batch_size: usize,
}

impl FusionOcrConfig {
    pub fn basic() -> Self {
        Self::default()
    }

    pub fn with_layout_confidence_threshold(mut self, threshold: f32) -> Self {
        self.layout_confidence_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    pub fn with_rec_confidence_threshold(mut self, threshold: f32) -> Self {
        self.rec_confidence_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    pub fn with_rec_batch_size(mut self, batch_size: usize) -> Self {
        self.rec_batch_size = batch_size.max(1);
        self
    }
}

impl Default for FusionOcrConfig {
    fn default() -> Self {
        Self {
            classes: LayoutClassConfig::basic(),
            layout_confidence_threshold: 0.4,
            rec_confidence_threshold: 0.0,
            rec_batch_size: 8,
        }
    }
}

/// 一个段落识别结果。
#[derive(Debug, Clone, PartialEq)]
pub struct FusionOcrParagraph {
    pub bbox: [f32; 4],
    pub paragraph_type: String,
    pub content: String,
}

/// 最近一次 `recognize` 各阶段耗时（未执行的阶段为 0）。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StageTiming {
    pub layout: Duration,
    pub detect: Duration,
    pub recognize: Duration,
}

/// 已加载模型的 OCR 引擎。可跨多张图片复用，避免重复加载 Session。
pub struct FusionOcr {
    config: FusionOcrConfig,
    layout: LayoutPredictor,
    text_detector: Option<TextDetector>,
    text_recognizer: Option<TextRecognizer>,
    formula_recognizer: Option<FormulaRecognizer>,
    last_timing: StageTiming,
}

impl FusionOcr {
    pub fn new(paths: FusionOcrModelPaths, config: FusionOcrConfig) -> FusionOcrResult<Self> {
        let layout = LayoutPredictor::with_threshold(
            &paths.layout_model,
            config.layout_confidence_threshold,
        )?;

        let text_detector = if config.classes.needs_det_model() {
            let det = required_path(&paths.text_det_model, "text_det_model")?;
            Some(TextDetector::new(det)?)
        } else {
            None
        };
        let text_recognizer = if config.classes.needs_rec_model() {
            let rec = required_path(&paths.text_rec_model, "text_rec_model")?;
            let dict = required_path(&paths.text_rec_dict, "text_rec_dict")?;
            Some(TextRecognizer::new(rec, dict)?.with_batch_size(config.rec_batch_size))
        } else {
            None
        };

        let formula_recognizer = if config.classes.formula {
            let model = required_path(&paths.formula_model, "formula_model")?;
            let tokens = required_path(&paths.formula_tokens, "formula_tokens")?;
            Some(FormulaRecognizer::new(model, tokens)?)
        } else {
            None
        };

        Ok(Self {
            config,
            layout,
            text_detector,
            text_recognizer,
            formula_recognizer,
            last_timing: StageTiming::default(),
        })
    }

    /// 最近一次 `recognize` 的各阶段耗时。
    pub fn last_timing(&self) -> StageTiming {
        self.last_timing
    }

    pub fn recognize(&mut self, image: &RgbImage) -> FusionOcrResult<Vec<FusionOcrParagraph>> {
        self.last_timing = StageTiming::default();
        // layout 与 det 相互独立（都以整页图像为输入），并行运行。
        // 两个 Session 各占一半逻辑核（见各构造函数），不会过度订阅。
        let layout = &mut self.layout;
        let text_detector = &mut self.text_detector;
        let scoped = std::thread::scope(|s| {
            let layout_handle = s.spawn(|| {
                let start = Instant::now();
                let result = layout.predict(image).map_err(|e| e.to_string());
                (result, start.elapsed())
            });
            let det_start = Instant::now();
            let lines = match text_detector.as_mut() {
                Some(detector) => detector.detect(image).map_err(|e| e.to_string()),
                None => Ok(Vec::new()),
            };
            let det_elapsed = det_start.elapsed();
            let (layout_result, layout_elapsed) = layout_handle
                .join()
                .map_err(|_| "layout thread panicked".to_string())?;
            Ok::<_, String>((layout_result?, lines?, layout_elapsed, det_elapsed))
        });
        let (layout_boxes, lines, layout_elapsed, det_elapsed) = scoped.map_err(PipelineError)?;
        self.last_timing.layout = layout_elapsed;
        self.last_timing.detect = det_elapsed;

        let mut regions: Vec<LayoutBox> = layout_boxes
            .into_iter()
            .filter(|region| self.config.classes.is_enabled(region.label))
            .collect();
        regions.sort_by(reading_order);

        let mut contents = vec![String::new(); regions.len()];
        self.recognize_text_regions(image, &regions, &lines, &mut contents)?;
        self.recognize_formula_regions(image, &regions, &mut contents)?;

        Ok(regions
            .into_iter()
            .enumerate()
            .map(|(index, region)| FusionOcrParagraph {
                bbox: region.bbox,
                paragraph_type: region.label.to_string(),
                content: contents[index].clone(),
            })
            .collect())
    }

    fn recognize_text_regions(
        &mut self,
        image: &RgbImage,
        regions: &[LayoutBox],
        lines: &[TextLine],
        contents: &mut [String],
    ) -> FusionOcrResult<()> {
        let text_regions: Vec<usize> = regions
            .iter()
            .enumerate()
            .filter(|(_, region)| is_text_label(region.label))
            .map(|(index, _)| index)
            .collect();
        if text_regions.is_empty() {
            return Ok(());
        }

        let recognizer = self
            .text_recognizer
            .as_mut()
            .expect("text models validated in new");
        let mut assigned: Vec<Vec<usize>> = (0..regions.len()).map(|_| Vec::new()).collect();
        for (line_index, line) in lines.iter().enumerate() {
            let cx = (line.bbox[0] + line.bbox[2]) / 2.0;
            let cy = (line.bbox[1] + line.bbox[3]) / 2.0;
            let target = text_regions
                .iter()
                .copied()
                .filter(|&index| inside(cx, cy, regions[index].bbox))
                .min_by(|&a, &b| {
                    area(regions[a].bbox)
                        .partial_cmp(&area(regions[b].bbox))
                        .unwrap()
                });
            if let Some(index) = target {
                assigned[index].push(line_index);
            }
        }

        let mut crops = Vec::new();
        let mut owners = Vec::new();
        for &region_index in &text_regions {
            let region = &regions[region_index];
            if is_single_line_label(region.label) || assigned[region_index].is_empty() {
                crops.push(crop(image, region.bbox));
                owners.push(region_index);
            } else {
                for &line_index in &assigned[region_index] {
                    crops.push(crop(image, lines[line_index].bbox));
                    owners.push(region_index);
                }
            }
        }

        let recognize_start = Instant::now();
        let recognized = recognizer.recognize_batch(&crops)?;
        self.last_timing.recognize = recognize_start.elapsed();
        let mut lines_by_region: Vec<Vec<String>> =
            (0..regions.len()).map(|_| Vec::new()).collect();
        for (owner, (text, confidence)) in owners.into_iter().zip(recognized) {
            if confidence >= self.config.rec_confidence_threshold && !text.trim().is_empty() {
                lines_by_region[owner].push(text.trim().to_string());
            }
        }
        for index in text_regions {
            contents[index] = markdown_text(regions[index].label, &lines_by_region[index]);
        }
        Ok(())
    }

    fn recognize_formula_regions(
        &mut self,
        image: &RgbImage,
        regions: &[LayoutBox],
        contents: &mut [String],
    ) -> FusionOcrResult<()> {
        let Some(recognizer) = self.formula_recognizer.as_mut() else {
            return Ok(());
        };
        for (index, region) in regions
            .iter()
            .enumerate()
            .filter(|(_, r)| r.label == "formula")
        {
            let latex = recognizer.recognize(&crop_with_padding(image, region.bbox, 8))?;
            if !latex.trim().is_empty() {
                contents[index] = format!("$$\n{}\n$$", latex.trim());
            }
        }
        Ok(())
    }
}

fn required_path<'a>(path: &'a Option<PathBuf>, name: &str) -> FusionOcrResult<&'a Path> {
    path.as_deref()
        .ok_or_else(|| PipelineError(format!("missing required model path: {name}")).into())
}

#[derive(Debug)]
struct PipelineError(String);
impl fmt::Display for PipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl Error for PipelineError {}

fn is_single_line_label(label: &str) -> bool {
    matches!(
        label,
        "paragraph_title"
            | "number"
            | "doc_title"
            | "footnote"
            | "header"
            | "algorithm"
            | "footer"
            | "formula_number"
    )
}

fn is_text_label(label: &str) -> bool {
    is_single_line_label(label) || is_multiline_text_label(label)
}

fn is_multiline_text_label(label: &str) -> bool {
    matches!(
        label,
        "text"
            | "abstract"
            | "content"
            | "figure_title"
            | "reference"
            | "aside_text"
            | "reference_content"
    )
}

fn markdown_text(label: &str, lines: &[String]) -> String {
    if lines.is_empty() {
        return String::new();
    }
    match label {
        "doc_title" => format!("# {}", lines.join(" ")),
        "paragraph_title" => format!("## {}", lines.join(" ")),
        _ => lines.join("\n"),
    }
}

fn reading_order(a: &LayoutBox, b: &LayoutBox) -> std::cmp::Ordering {
    a.bbox[1]
        .partial_cmp(&b.bbox[1])
        .unwrap()
        .then(a.bbox[0].partial_cmp(&b.bbox[0]).unwrap())
}

fn crop(image: &RgbImage, bbox: [f32; 4]) -> RgbImage {
    crop_with_padding(image, bbox, 0)
}
fn crop_with_padding(image: &RgbImage, bbox: [f32; 4], padding: u32) -> RgbImage {
    let x1 = (bbox[0] as i32 - padding as i32).max(0) as u32;
    let y1 = (bbox[1] as i32 - padding as i32).max(0) as u32;
    let x2 = (bbox[2] as i32 + padding as i32).min(image.width() as i32) as u32;
    let y2 = (bbox[3] as i32 + padding as i32).min(image.height() as i32) as u32;
    image::imageops::crop_imm(image, x1, y1, (x2 - x1).max(1), (y2 - y1).max(1)).to_image()
}
fn inside(x: f32, y: f32, bbox: [f32; 4]) -> bool {
    x >= bbox[0] && x <= bbox[2] && y >= bbox[1] && y <= bbox[3]
}
fn area(bbox: [f32; 4]) -> f32 {
    (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_enables_core_labels() {
        let classes = LayoutClassConfig::basic();
        for label in LABELS {
            assert_eq!(
                classes.is_enabled(label),
                matches!(
                    label,
                    "paragraph_title" | "text" | "doc_title" | "abstract" | "figure_title"
                )
            );
        }
    }

    #[test]
    fn default_thresholds_and_batch_size() {
        let config = FusionOcrConfig::default();
        assert_eq!(config.layout_confidence_threshold, 0.4);
        assert_eq!(config.rec_confidence_threshold, 0.0);
        assert_eq!(config.rec_batch_size, 8);
    }

    #[test]
    fn markdown_formats_titles() {
        assert_eq!(markdown_text("doc_title", &["Hello".into()]), "# Hello");
        assert_eq!(
            markdown_text("paragraph_title", &["Hello".into()]),
            "## Hello"
        );
        assert_eq!(markdown_text("text", &["a".into(), "b".into()]), "a\nb");
    }

    #[test]
    fn builders_clamp_invalid_values() {
        let config = FusionOcrConfig::basic()
            .with_layout_confidence_threshold(1.5)
            .with_rec_confidence_threshold(-0.5)
            .with_rec_batch_size(0);
        assert_eq!(config.layout_confidence_threshold, 1.0);
        assert_eq!(config.rec_confidence_threshold, 0.0);
        assert_eq!(config.rec_batch_size, 1);
    }

    #[test]
    fn classes_can_be_changed_by_label() {
        let classes = LayoutClassConfig::basic()
            .with_enabled("text", false)
            .with_enabled("formula", true)
            .with_enabled("abstract", true);
        assert!(!classes.text);
        assert!(classes.formula);
        assert!(classes.r#abstract);
    }

    #[test]
    fn single_line_only_does_not_need_det() {
        let classes = LayoutClassConfig::none().with_enabled("paragraph_title", true);
        assert!(classes.needs_rec_model());
        assert!(!classes.needs_det_model());
    }

    #[test]
    fn figure_title_is_recognized_when_enabled() {
        // 图注可能跨多行（如 "Figure 1. ..."），需要 det 逐行检测后再 rec。
        let classes = LayoutClassConfig::none().with_enabled("figure_title", true);
        assert!(classes.needs_rec_model());
        assert!(classes.needs_det_model());
    }

    #[test]
    fn formula_does_not_enable_text_models() {
        let classes = LayoutClassConfig::none().with_enabled("formula", true);
        assert!(!classes.needs_rec_model());
        assert!(!classes.needs_det_model());
    }
}

//! 公式识别，基于 PP-FormulaNet_plus-S（UniMERNet 架构，Donut 编码器-解码器）。
//! greedy search 已内嵌在计算图中（Loop 算子），单次推理即输出完整 token 序列。
//! 解码为 byte-level BPE 逆变换，词表文件由 `scripts/extract_unimernet_tokens.py` 生成。

use image::{RgbImage, imageops::FilterType};
use ndarray::Array4;
use ort::{inputs, session::Session, session::builder::GraphOptimizationLevel, value::Tensor};
use std::{error::Error, path::Path};

/// 模型输入边长（ONNX 签名 [B, 1, 384, 384]，单通道灰度）。
const INPUT_SIZE: usize = 384;
/// `</eos_token>`，greedy search 结束标记。
const EOS_TOKEN: i64 = 2;
/// id 0..=22 为特殊 token（见 yml added_tokens）。
const NUM_SPECIAL_TOKENS: i64 = 23;
/// 该 ONNX 的内嵌 Loop 按 batch 扩展解码循环后 GPU 效率下降，实测单公式推理更快。
const BATCH_SIZE: usize = 1;

/// 公式识别器，将公式图像识别为 LaTeX 字符串。
pub struct FormulaRecognizer {
    session: Session,
    /// 词表，索引即 token id。
    tokens: Vec<String>,
    /// 归一化参数（对 /255 后的灰度值做 (v - mean) / std）。
    mean: f32,
    std: f32,
}

impl FormulaRecognizer {
    /// 从 ONNX 模型文件与词表文件创建识别器。
    pub fn new<P: AsRef<Path>>(model_path: P, tokens_path: P) -> Result<Self, Box<dyn Error>> {
        let session = crate::ocr::cpu_session_builder()?
            .with_optimization_level(GraphOptimizationLevel::All)?
            .with_intra_threads(8)?
            .commit_from_file(model_path)?;
        let tokens = std::fs::read_to_string(tokens_path)?
            .lines()
            .map(str::to_owned)
            .collect();
        Ok(Self {
            session,
            tokens,
            // PaddleOCR UniMERNetTestTransform 官方参数。
            mean: 0.7931,
            std: 0.1738,
        })
    }

    /// 指定归一化参数（默认 mean=0.7931, std=0.1738）。
    pub fn with_normalization(mut self, mean: f32, std: f32) -> Self {
        self.mean = mean;
        self.std = std;
        self
    }

    /// 识别一张公式图像，返回 LaTeX 字符串。
    pub fn recognize(&mut self, img: &RgbImage) -> Result<String, Box<dyn Error>> {
        Ok(self.recognize_batch(std::slice::from_ref(img))?.remove(0))
    }

    /// 批量识别公式，按输入顺序返回 LaTeX。
    pub fn recognize_batch(&mut self, imgs: &[RgbImage]) -> Result<Vec<String>, Box<dyn Error>> {
        self.recognize_batch_with_size(imgs, BATCH_SIZE)
    }

    /// 指定 micro-batch 大小。FormulaNet 含自回归 Loop，建议仅对复杂度相近的公式使用 2。
    pub fn recognize_batch_with_size(
        &mut self,
        imgs: &[RgbImage],
        batch_size: usize,
    ) -> Result<Vec<String>, Box<dyn Error>> {
        let mut result = Vec::with_capacity(imgs.len());
        for batch in imgs.chunks(batch_size.max(1)) {
            let mut data = Vec::with_capacity(batch.len() * INPUT_SIZE * INPUT_SIZE);
            for img in batch {
                data.extend(self.preprocess(img));
            }
            let x = Array4::from_shape_vec((batch.len(), 1, INPUT_SIZE, INPUT_SIZE), data)?;
            let outputs = self.session.run(inputs!["x" => Tensor::from_array(x)?])?;
            let (shape, ids) = outputs["fetch_name_0"].try_extract_tensor::<i64>()?;
            if shape.len() != 2 || shape[0] as usize != batch.len() {
                return Err(format!("unexpected formula output shape: {shape:?}").into());
            }
            let sequence_len = shape[1] as usize;
            for row in ids.chunks_exact(sequence_len) {
                result.push(Self::decode(&self.tokens, row));
            }
        }
        Ok(result)
    }

    fn preprocess(&self, img: &RgbImage) -> Vec<f32> {
        // UniMERNetImgDecode: 裁掉空白边缘，保持纵横比缩放，白底居中填充到 384x384。
        let gray = image::imageops::grayscale(img);
        let gray = crop_margin(&gray);
        let scale =
            (INPUT_SIZE as f32 / gray.width() as f32).min(INPUT_SIZE as f32 / gray.height() as f32);
        let rw = ((gray.width() as f32 * scale).round() as u32).clamp(1, INPUT_SIZE as u32);
        let rh = ((gray.height() as f32 * scale).round() as u32).clamp(1, INPUT_SIZE as u32);
        let resized = image::imageops::resize(&gray, rw, rh, FilterType::Triangle);
        let mut canvas =
            image::GrayImage::from_pixel(INPUT_SIZE as u32, INPUT_SIZE as u32, image::Luma([255]));
        image::imageops::replace(
            &mut canvas,
            &resized,
            ((INPUT_SIZE as u32 - rw) / 2) as i64,
            ((INPUT_SIZE as u32 - rh) / 2) as i64,
        );

        // UniMERNetTestTransform: ToGray + Normalize(0.7931, 0.1738)。
        let mut data = vec![0.0f32; INPUT_SIZE * INPUT_SIZE];
        for (i, p) in canvas.pixels().enumerate() {
            data[i] = (p[0] as f32 / 255.0 - self.mean) / self.std;
        }
        data
    }

    /// token id 序列 -> LaTeX：查表拼接 -> byte-level 逆变换 -> UTF-8。
    fn decode(tokens: &[String], ids: &[i64]) -> String {
        let mut buf = String::new();
        for &id in ids {
            if id == EOS_TOKEN {
                break;
            }
            if id < NUM_SPECIAL_TOKENS {
                continue;
            }
            if let Some(t) = tokens.get(id as usize) {
                buf.push_str(t);
            }
        }
        let bytes: Vec<u8> = buf.chars().map(byte_level_decode_char).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// 与 PaddleOCR UniMERNetImgDecode.crop_margin 一致：对比度归一化后取灰度 < 200 的前景包围框。
fn crop_margin(gray: &image::GrayImage) -> image::GrayImage {
    let mut min = u8::MAX;
    let mut max = u8::MIN;
    for p in gray.pixels() {
        min = min.min(p[0]);
        max = max.max(p[0]);
    }
    if min == max {
        return gray.clone();
    }

    let (mut x1, mut y1) = (gray.width(), gray.height());
    let (mut x2, mut y2) = (0u32, 0u32);
    let mut found = false;
    for (x, y, p) in gray.enumerate_pixels() {
        let normalized = (p[0] as f32 - min as f32) / (max as f32 - min as f32) * 255.0;
        if normalized < 200.0 {
            x1 = x1.min(x);
            y1 = y1.min(y);
            x2 = x2.max(x);
            y2 = y2.max(y);
            found = true;
        }
    }
    if !found || x2 < x1 || y2 < y1 {
        return gray.clone();
    }
    image::imageops::crop_imm(gray, x1, y1, x2 - x1 + 1, y2 - y1 + 1).to_image()
}

/// byte-level BPE 的 bytes_to_unicode 逆映射：可打印字节（33-126, 161-172, 174-255）
/// 映射为自身，其余字节按序映射到 256+n，此处做逆变换。
fn byte_level_decode_char(c: char) -> u8 {
    let cp = c as u32;
    if cp < 256 {
        return cp as u8;
    }
    let n = cp - 256;
    let mut count = 0;
    for b in 0..=255u32 {
        let printable = matches!(b, 33..=126 | 161..=172 | 174..=255);
        if !printable {
            if count == n {
                return b as u8;
            }
            count += 1;
        }
    }
    0
}

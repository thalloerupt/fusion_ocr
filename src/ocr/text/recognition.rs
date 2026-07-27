//! 单行文字识别，基于 PP-OCRv6_tiny_rec（CTC 解码）。
//! 预处理/后处理与 `models/PP-OCRv6_tiny_rec.yml` 保持一致。

use image::{RgbImage, imageops::FilterType};
use ndarray::Array4;
use ort::{inputs, session::Session, value::Tensor};
use std::{error::Error, path::Path};

/// 模型输入高度（yml: RecResizeImg image_shape [3, 48, 320]）。
const INPUT_HEIGHT: u32 = 48;
/// 输入宽度上限（yml: trt_dynamic_shapes 最大 3200）。
const MAX_INPUT_WIDTH: u32 = 3200;
/// 批量识别时每批最大图片数（同 PaddleOCR rec_batch_num 惯例）。
const DEFAULT_BATCH_SIZE: usize = 8;

/// 文字识别器，按行/区域图像逐张或批量识别。
pub struct TextRecognizer {
    session: Session,
    compact_output: bool,
    batch_size: usize,
    /// 字典，索引 i 对应模型输出类别 i + 1（类别 0 为 CTC blank）。
    dict: Vec<String>,
}

impl TextRecognizer {
    /// 从 ONNX 模型文件与字典文件创建识别器。
    pub fn new<P: AsRef<Path>>(model_path: P, dict_path: P) -> Result<Self, Box<dyn Error>> {
        // rec 独占阶段运行，占满全部逻辑核。
        let session = crate::ocr::session_builder(crate::ocr::available_threads())?
            .commit_from_file(model_path)?;
        let compact_output = session.outputs().iter().any(|o| o.name() == "token_ids");
        let dict = std::fs::read_to_string(dict_path)?
            .lines()
            .map(str::to_owned)
            .collect();
        Ok(Self {
            session,
            compact_output,
            batch_size: DEFAULT_BATCH_SIZE,
            dict,
        })
    }

    /// 设置批量识别大小。输入仍会按宽度排序以降低 padding 浪费。
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size.max(1);
        self
    }

    /// 识别一张文字区域图像，返回 `(识别文本, 平均置信度)`。
    pub fn recognize(&mut self, img: &RgbImage) -> Result<(String, f32), Box<dyn Error>> {
        if img.width() == 0 || img.height() == 0 {
            return Ok((String::new(), 0.0));
        }
        let (w, data) = Self::resize_norm(img);
        let x = Array4::from_shape_vec((1, 3, INPUT_HEIGHT as usize, w), data)?;
        let outputs = self.session.run(inputs!["x" => Tensor::from_array(x)?])?;
        if self.compact_output {
            let (_, ids) = outputs["token_ids"].try_extract_tensor::<i64>()?;
            let (_, probs) = outputs["token_probs"].try_extract_tensor::<f32>()?;
            Ok(Self::ctc_decode_compact(&self.dict, ids, probs))
        } else {
            let (shape, logits) = outputs["fetch_name_0"].try_extract_tensor::<f32>()?;
            Ok(Self::ctc_decode(
                &self.dict,
                logits,
                shape[1] as usize,
                shape[2] as usize,
            ))
        }
    }

    /// 批量识别多张文字区域图像，按输入顺序返回 `(识别文本, 平均置信度)`。
    ///
    /// 图像按缩放后宽度升序分组，组内拼成一个 batch 推理，摊薄调用开销。
    pub fn recognize_batch(
        &mut self,
        imgs: &[RgbImage],
    ) -> Result<Vec<(String, f32)>, Box<dyn Error>> {
        let mut results = vec![(String::new(), 0.0f32); imgs.len()];
        // 预处理，空图直接返回空结果
        let mut prepared = Vec::with_capacity(imgs.len());
        let mut order = Vec::with_capacity(imgs.len());
        for (i, img) in imgs.iter().enumerate() {
            if img.width() == 0 || img.height() == 0 {
                prepared.push(None);
            } else {
                prepared.push(Some(Self::resize_norm(img)));
                order.push(i);
            }
        }
        // 按宽度升序分组，减少组内 padding 浪费
        order.sort_by_key(|&i| prepared[i].as_ref().unwrap().0);

        for group in order.chunks(self.batch_size) {
            let h = INPUT_HEIGHT as usize;
            let w_max = group
                .iter()
                .map(|&i| prepared[i].as_ref().unwrap().0)
                .max()
                .unwrap();
            let n = group.len();
            let mut buf = vec![0.0f32; n * 3 * h * w_max];
            for (bi, &i) in group.iter().enumerate() {
                let (w, data) = prepared[i].as_ref().unwrap();
                for c in 0..3 {
                    for row in 0..h {
                        let src = (c * h + row) * w;
                        let dst = ((bi * 3 + c) * h + row) * w_max;
                        buf[dst..dst + w].copy_from_slice(&data[src..src + w]);
                    }
                }
            }
            let x = Array4::from_shape_vec((n, 3, h, w_max), buf)?;
            let outputs = self.session.run(inputs!["x" => Tensor::from_array(x)?])?;
            if self.compact_output {
                let (shape, ids) = outputs["token_ids"].try_extract_tensor::<i64>()?;
                let (_, probs) = outputs["token_probs"].try_extract_tensor::<f32>()?;
                let steps = shape[1] as usize;
                for (bi, &i) in group.iter().enumerate() {
                    results[i] = Self::ctc_decode_compact(
                        &self.dict,
                        &ids[bi * steps..(bi + 1) * steps],
                        &probs[bi * steps..(bi + 1) * steps],
                    );
                }
            } else {
                let (shape, logits) = outputs["fetch_name_0"].try_extract_tensor::<f32>()?;
                let (steps, num_classes) = (shape[1] as usize, shape[2] as usize);
                for (bi, &i) in group.iter().enumerate() {
                    let row = &logits[bi * steps * num_classes..(bi + 1) * steps * num_classes];
                    results[i] = Self::ctc_decode(&self.dict, row, steps, num_classes);
                }
            }
        }
        Ok(results)
    }

    /// RecResizeImg + Normalize：按宽高比缩放到高 48（宽上限 MAX_INPUT_WIDTH），
    /// /255 -> (x-0.5)/0.5，BGR 通道序，返回 `(宽, CHW 数据)`。
    fn resize_norm(img: &RgbImage) -> (usize, Vec<f32>) {
        let (w, h) = (img.width(), img.height());
        let resized_w =
            ((INPUT_HEIGHT as f32 * w as f32 / h as f32).round() as u32).clamp(1, MAX_INPUT_WIDTH);
        let resized = image::imageops::resize(img, resized_w, INPUT_HEIGHT, FilterType::Triangle);
        let (rw, rh) = (resized_w as usize, INPUT_HEIGHT as usize);
        let plane = rh * rw;
        let mut data = vec![0.0f32; 3 * plane];
        for (i, px) in resized.as_raw().chunks_exact(3).enumerate() {
            data[i] = (px[2] as f32 / 255.0 - 0.5) / 0.5; // B
            data[plane + i] = (px[1] as f32 / 255.0 - 0.5) / 0.5; // G
            data[2 * plane + i] = (px[0] as f32 / 255.0 - 0.5) / 0.5; // R
        }
        (rw, data)
    }

    /// CTCLabelDecode：argmax -> 合并连续重复 -> 去 blank(0)，返回 `(文本, 平均置信度)`。
    fn ctc_decode(
        dict: &[String],
        logits: &[f32],
        steps: usize,
        num_classes: usize,
    ) -> (String, f32) {
        let mut text = String::new();
        let (mut conf_sum, mut conf_cnt) = (0.0f32, 0usize);
        let mut prev = 0usize;
        for i in 0..steps {
            let row = &logits[i * num_classes..(i + 1) * num_classes];
            let mut idx = 0usize;
            let mut max_prob = f32::NEG_INFINITY;
            for (j, &v) in row.iter().enumerate() {
                if v > max_prob {
                    max_prob = v;
                    idx = j;
                }
            }
            if idx != 0 && idx != prev {
                if let Some(ch) = dict.get(idx - 1) {
                    text.push_str(ch);
                    // 模型输出已经是 softmax 概率分布，直接取最大值
                    conf_sum += max_prob;
                    conf_cnt += 1;
                }
            }
            prev = idx;
        }
        let conf = if conf_cnt > 0 {
            conf_sum / conf_cnt as f32
        } else {
            0.0
        };
        (text, conf)
    }

    fn ctc_decode_compact(dict: &[String], ids: &[i64], probs: &[f32]) -> (String, f32) {
        let mut text = String::new();
        let (mut conf_sum, mut conf_cnt) = (0.0f32, 0usize);
        let mut prev = 0i64;
        for (&id, &prob) in ids.iter().zip(probs) {
            if id != 0 && id != prev {
                if let Some(ch) = dict.get(id as usize - 1) {
                    text.push_str(ch);
                    conf_sum += prob;
                    conf_cnt += 1;
                }
            }
            prev = id;
        }
        let conf = if conf_cnt > 0 {
            conf_sum / conf_cnt as f32
        } else {
            0.0
        };
        (text, conf)
    }
}

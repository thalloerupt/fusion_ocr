//! 文本行检测，基于 PP-OCRv6_tiny_det（DB 算法）。
//! 预处理/后处理与 `models/PP-OCRv6_tiny_det.yml` 保持一致。

use image::{GrayImage, Luma, RgbImage, imageops::FilterType};
use imageproc::{drawing::draw_polygon_mut, point::Point};
use ndarray::Array4;
use ort::{inputs, session::Session, session::builder::GraphOptimizationLevel, value::Tensor};
use std::{error::Error, path::Path};

/// DetResizeForTest 短边长度（limit_type = "min"）。
const LIMIT_SIDE_LEN: u32 = 736;
/// 缩放后长边上界（与 yml 中 trt_dynamic_shapes 上限一致）。
const MAX_SIDE_LEN: u32 = 4000;
/// 默认长边上限：大图先按比例缩小再送检，CPU 上推理/后处理成倍加速。
/// 比 PaddleOCR 官方默认 960 保守，降低小字漏检风险。
const DEFAULT_MAX_DET_SIDE: u32 = 1280;
/// NormalizeImage 参数（yml）。
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// 检测到的一行文本。
#[derive(Debug, Clone)]
pub struct TextLine {
    /// 轴对齐外接框 `[x1, y1, x2, y2]`，输入图像素坐标，已裁剪到图像范围内。
    pub bbox: [f32; 4],
    /// 置信度（四边形内平均概率）。
    pub score: f32,
}

/// 文本行检测器。
pub struct TextDetector {
    session: Session,
    /// 二值化阈值（yml: DBPostProcess thresh）。
    thresh: f32,
    /// 框平均概率阈值（yml: box_thresh）。
    box_thresh: f32,
    /// 框扩张比例（yml: unclip_ratio）。
    unclip_ratio: f32,
    /// 最大候选框数（yml: max_candidates）。
    max_candidates: usize,
    /// 最短边下限（DBPostProcess 默认 3）。
    min_box_size: f32,
    /// 送检图像长边上限。
    max_det_side: u32,
}

impl TextDetector {
    /// 从 ONNX 模型文件创建检测器，后处理参数取自 yml。
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self, Box<dyn Error>> {
        // det 与 layout 并行运行（见 pipeline），各占一半逻辑核，避免过度订阅。
        let session = crate::ocr::session_builder((crate::ocr::available_threads() / 2).max(1))?
            .with_optimization_level(GraphOptimizationLevel::All)?
            .commit_from_file(model_path)?;
        Ok(Self {
            session,
            thresh: 0.2,
            box_thresh: 0.4,
            unclip_ratio: 1.4,
            max_candidates: 3000,
            min_box_size: 3.0,
            max_det_side: DEFAULT_MAX_DET_SIDE,
        })
    }

    /// 设置送检图像长边上限（默认 [`DEFAULT_MAX_DET_SIDE`]）。
    /// 调大可提升大图中小字的召回，调小则更快。
    pub fn with_max_det_side(mut self, max_det_side: u32) -> Self {
        self.max_det_side = max_det_side.clamp(LIMIT_SIDE_LEN, MAX_SIDE_LEN);
        self
    }

    /// 检测图像中的文本行，按阅读顺序（自上而下、自左而右）返回。
    pub fn detect(&mut self, img: &RgbImage) -> Result<Vec<TextLine>, Box<dyn Error>> {
        let (orig_w, orig_h) = (img.width(), img.height());
        if orig_w == 0 || orig_h == 0 {
            return Ok(Vec::new());
        }

        // Preprocess: DetResizeForTest（短边对齐 LIMIT_SIDE_LEN，长边不超过 max_det_side，边长取 32 的倍数）
        let (resized_w, resized_h) = resize_for_test(orig_w, orig_h, self.max_det_side);
        let resized = image::imageops::resize(img, resized_w, resized_h, FilterType::Triangle);

        // NormalizeImage(BGR, mean/std) -> ToCHWImage
        let (rw, rh) = (resized_w as usize, resized_h as usize);
        let mut x = Array4::<f32>::zeros((1, 3, rh, rw));
        let plane = rh * rw;
        let buf = x.as_slice_mut().expect("Array4::zeros 是连续内存");
        for (i, px) in resized.as_raw().chunks_exact(3).enumerate() {
            buf[i] = (px[2] as f32 / 255.0 - MEAN[0]) / STD[0]; // B
            buf[plane + i] = (px[1] as f32 / 255.0 - MEAN[1]) / STD[1]; // G
            buf[2 * plane + i] = (px[0] as f32 / 255.0 - MEAN[2]) / STD[2]; // R
        }

        let outputs = self.session.run(inputs!["x" => Tensor::from_array(x)?])?;

        // Postprocess: DBPostProcess
        let (shape, pred) = outputs["fetch_name_0"].try_extract_tensor::<f32>()?;
        let (ph, pw) = (shape[2] as usize, shape[3] as usize);

        // 二值化（外加 1px 零边框，保证贴边区域也能提取到轮廓）
        let mut mask = GrayImage::new(pw as u32 + 2, ph as u32 + 2);
        for row in 0..ph {
            for col in 0..pw {
                if pred[row * pw + col] > self.thresh {
                    mask.put_pixel(col as u32 + 1, row as u32 + 1, Luma([255]));
                }
            }
        }

        let mut lines = Vec::new();
        for contour in imageproc::contours::find_contours::<i32>(&mask)
            .into_iter()
            .take(self.max_candidates)
        {
            let points: Vec<(f32, f32)> = contour
                .points
                .iter()
                .map(|p| (p.x as f32 - 1.0, p.y as f32 - 1.0))
                .collect();
            if points.len() < 4 {
                continue;
            }
            let quad = min_area_quad(&points);
            if short_side(&quad) < self.min_box_size {
                continue;
            }
            let score = quad_score(pred, pw, ph, &quad);
            if score < self.box_thresh {
                continue;
            }
            let quad = unclip(&quad, self.unclip_ratio);
            if short_side(&quad) < self.min_box_size + 2.0 {
                continue;
            }
            // 映射回原图坐标
            let (sx, sy) = (orig_w as f32 / pw as f32, orig_h as f32 / ph as f32);
            let xs: Vec<f32> = quad
                .iter()
                .map(|p| (p.0 * sx).clamp(0.0, orig_w as f32))
                .collect();
            let ys: Vec<f32> = quad
                .iter()
                .map(|p| (p.1 * sy).clamp(0.0, orig_h as f32))
                .collect();
            let (x1, x2) = (minf(&xs), maxf(&xs));
            let (y1, y2) = (minf(&ys), maxf(&ys));
            if x2 - x1 < 1.0 || y2 - y1 < 1.0 {
                continue;
            }
            lines.push(TextLine {
                bbox: [x1, y1, x2, y2],
                score,
            });
        }

        sort_reading_order(&mut lines);
        Ok(lines)
    }
}

/// 按阅读顺序排序：先按 (y, x) 排序，再将 y 坐标接近（同一基线）的相邻框按 x 交换，
/// 与 PaddleOCR 的 `sorted_boxes` 一致（y 阈值 10px）。
fn sort_reading_order(lines: &mut [TextLine]) {
    lines.sort_by(|a, b| {
        a.bbox[1]
            .partial_cmp(&b.bbox[1])
            .unwrap()
            .then(a.bbox[0].partial_cmp(&b.bbox[0]).unwrap())
    });
    for i in 0..lines.len().saturating_sub(1) {
        for j in (0..=i).rev() {
            if (lines[j + 1].bbox[1] - lines[j].bbox[1]).abs() < 10.0
                && lines[j + 1].bbox[0] < lines[j].bbox[0]
            {
                lines.swap(j, j + 1);
            } else {
                break;
            }
        }
    }
}

/// DetResizeForTest（limit_type = "min"）：短边不足 LIMIT_SIDE_LEN 时放大，
/// 长边超过 max_det_side 时缩小，边长对齐 32。
fn resize_for_test(w: u32, h: u32, max_det_side: u32) -> (u32, u32) {
    let mut ratio = if w.min(h) < LIMIT_SIDE_LEN {
        LIMIT_SIDE_LEN as f32 / w.min(h) as f32
    } else {
        1.0
    };
    if (w.max(h) as f32 * ratio) > max_det_side as f32 {
        ratio = max_det_side as f32 / w.max(h) as f32;
    }
    let mut rw = ((w as f32 * ratio) / 32.0).round().max(1.0) as u32 * 32;
    let mut rh = ((h as f32 * ratio) / 32.0).round().max(1.0) as u32 * 32;
    if rw.max(rh) > MAX_SIDE_LEN {
        let scale = MAX_SIDE_LEN as f32 / rw.max(rh) as f32;
        rw = ((rw as f32 * scale) as u32 / 32).max(1) * 32;
        rh = ((rh as f32 * scale) as u32 / 32).max(1) * 32;
    }
    (rw, rh)
}

fn minf(v: &[f32]) -> f32 {
    v.iter().cloned().fold(f32::INFINITY, f32::min)
}

fn maxf(v: &[f32]) -> f32 {
    v.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
}

fn cross(o: (f32, f32), a: (f32, f32), b: (f32, f32)) -> f32 {
    (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
}

/// 单调链凸包，返回逆时针顶点（不重复首尾点）。
fn convex_hull(points: &[(f32, f32)]) -> Vec<(f32, f32)> {
    let mut pts = points.to_vec();
    pts.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap()
            .then(a.1.partial_cmp(&b.1).unwrap())
    });
    pts.dedup();
    if pts.len() <= 1 {
        return pts;
    }
    let mut hull: Vec<(f32, f32)> = Vec::with_capacity(pts.len() * 2);
    for &p in &pts {
        while hull.len() >= 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }
    let lower_len = hull.len();
    for &p in pts.iter().rev().skip(1) {
        while hull.len() > lower_len && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0
        {
            hull.pop();
        }
        hull.push(p);
    }
    hull.pop();
    hull
}

/// 旋转卡壳求最小外接矩形，返回四角点（沿凸包边序，保证绕向一致）。
fn min_area_quad(points: &[(f32, f32)]) -> [(f32, f32); 4] {
    let hull = convex_hull(points);
    let n = hull.len();
    if n == 0 {
        return [(0.0, 0.0); 4];
    }
    if n < 3 {
        let (a, b) = (hull[0], hull[n - 1]);
        return [a, b, b, a];
    }
    let mut best_area = f32::INFINITY;
    let mut best = [(0.0, 0.0); 4];
    for i in 0..n {
        let (x1, y1) = hull[i];
        let (x2, y2) = hull[(i + 1) % n];
        let (dx, dy) = (x2 - x1, y2 - y1);
        let len = dx.hypot(dy);
        if len < 1e-6 {
            continue;
        }
        let (ux, uy) = (dx / len, dy / len);
        let (nx, ny) = (-uy, ux);
        let (mut min_u, mut max_u) = (f32::INFINITY, f32::NEG_INFINITY);
        let (mut min_v, mut max_v) = (f32::INFINITY, f32::NEG_INFINITY);
        for &(px, py) in &hull {
            let u = px * ux + py * uy;
            let v = px * nx + py * ny;
            min_u = min_u.min(u);
            max_u = max_u.max(u);
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
        let area = (max_u - min_u) * (max_v - min_v);
        if area < best_area {
            best_area = area;
            let to_xy = |u: f32, v: f32| (u * ux + v * nx, u * uy + v * ny);
            best = [
                to_xy(min_u, min_v),
                to_xy(max_u, min_v),
                to_xy(max_u, max_v),
                to_xy(min_u, max_v),
            ];
        }
    }
    best
}

/// 四边形最短边长。
fn short_side(quad: &[(f32, f32); 4]) -> f32 {
    (0..4)
        .map(|i| {
            let (x1, y1) = quad[i];
            let (x2, y2) = quad[(i + 1) % 4];
            (x2 - x1).hypot(y2 - y1)
        })
        .fold(f32::INFINITY, f32::min)
}

/// 四边形绕向（正为逆时针）。
fn signed_area(quad: &[(f32, f32); 4]) -> f32 {
    (0..4)
        .map(|i| {
            let (x1, y1) = quad[i];
            let (x2, y2) = quad[(i + 1) % 4];
            x1 * y2 - x2 * y1
        })
        .sum::<f32>()
        / 2.0
}

/// DBPostProcess 的 box_score_fast：四边形掩码内的平均概率。
fn quad_score(pred: &[f32], pw: usize, ph: usize, quad: &[(f32, f32); 4]) -> f32 {
    let xs: Vec<f32> = quad.iter().map(|p| p.0).collect();
    let ys: Vec<f32> = quad.iter().map(|p| p.1).collect();
    let xmin = minf(&xs).floor().clamp(0.0, pw as f32 - 1.0) as i32;
    let xmax = maxf(&xs).ceil().clamp(0.0, pw as f32 - 1.0) as i32;
    let ymin = minf(&ys).floor().clamp(0.0, ph as f32 - 1.0) as i32;
    let ymax = maxf(&ys).ceil().clamp(0.0, ph as f32 - 1.0) as i32;
    let (bw, bh) = ((xmax - xmin + 1) as u32, (ymax - ymin + 1) as u32);
    let mut mask = GrayImage::new(bw, bh);
    let poly: Vec<Point<i32>> = quad
        .iter()
        .map(|p| {
            Point::new(
                (p.0 - xmin as f32).round() as i32,
                (p.1 - ymin as f32).round() as i32,
            )
        })
        .collect();
    draw_polygon_mut(&mut mask, &poly, Luma([255]));
    let (mut sum, mut cnt) = (0.0f32, 0u32);
    for row in 0..bh {
        for col in 0..bw {
            if mask.get_pixel(col, row)[0] > 0 {
                sum += pred[(ymin as usize + row as usize) * pw + xmin as usize + col as usize];
                cnt += 1;
            }
        }
    }
    if cnt == 0 { 0.0 } else { sum / cnt as f32 }
}

/// DBPostProcess 的 unclip：凸四边形沿外法线扩张 `面积 * ratio / 周长`。
fn unclip(quad: &[(f32, f32); 4], ratio: f32) -> [(f32, f32); 4] {
    let mut q = *quad;
    if signed_area(&q) < 0.0 {
        q.swap(1, 3); // 翻转为逆时针，外法线方向统一
    }
    let area = signed_area(&q).abs();
    let perimeter: f32 = (0..4)
        .map(|i| {
            let (x1, y1) = q[i];
            let (x2, y2) = q[(i + 1) % 4];
            (x2 - x1).hypot(y2 - y1)
        })
        .sum();
    if perimeter < 1e-6 {
        return *quad;
    }
    let d = area * ratio / perimeter;
    // 每条边沿外法线平移 d，新顶点为相邻平移边的交点
    let mut edges = [((0.0, 0.0), (0.0, 0.0)); 4]; // (偏移后边上一点, 方向)
    for i in 0..4 {
        let (x1, y1) = q[i];
        let (x2, y2) = q[(i + 1) % 4];
        let (dx, dy) = (x2 - x1, y2 - y1);
        let len = dx.hypot(dy);
        if len < 1e-6 {
            return *quad;
        }
        let (nx, ny) = (dy / len, -dx / len); // 逆时针多边形的外法线
        edges[i] = ((x1 + nx * d, y1 + ny * d), (dx / len, dy / len));
    }
    let mut out = *quad;
    for i in 0..4 {
        let (p1, d1) = edges[(i + 3) % 4];
        let (p2, d2) = edges[i];
        let cross = d1.0 * d2.1 - d1.1 * d2.0;
        if cross.abs() < 1e-8 {
            return *quad;
        }
        let t = ((p2.0 - p1.0) * d2.1 - (p2.1 - p1.1) * d2.0) / cross;
        out[i] = (p1.0 + t * d1.0, p1.1 + t * d1.1);
    }
    out
}

"""Deterministic HSV candidate segmentation. RGB input; all geometry in analysis pixels."""
from dataclasses import dataclass
from io import BytesIO
from time import perf_counter
import cv2
import numpy as np
from PIL import Image, ImageOps


def load_image(data: bytes) -> np.ndarray:
    if len(data) > 20 * 1024 * 1024:
        raise ValueError('20 MB 이하의 이미지를 사용해 주세요.')
    with Image.open(BytesIO(data)) as im:
        if im.width * im.height > 25_000_000:
            raise ValueError('이미지는 2,500만 픽셀 이하여야 합니다.')
        return np.array(ImageOps.exif_transpose(im).convert('RGB'))


def resize_image(rgb, max_side=1200):
    h, w = rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    return cv2.resize(rgb, (max(1, round(w*scale)), max(1, round(h*scale))), interpolation=cv2.INTER_AREA) if scale < 1 else rgb.copy()


def color_masks(hsv):
    h, s, v = cv2.split(hsv)
    dark = v < 60
    red = ((h <= 10) | (h >= 170)) & (s >= 60) & ~dark
    return red, dark


def color_ratios(hsv, selected=None):
    red, dark = color_masks(hsv)
    selected = np.ones(red.shape, bool) if selected is None else selected.astype(bool)
    n = int(selected.sum())
    if not n:
        return dict(red=0., dark=0., other=0.)
    r, d = 100*np.count_nonzero(red & selected)/n, 100*np.count_nonzero(dark & selected)/n
    return dict(red=float(r), dark=float(d), other=float(100-r-d))


@dataclass
class AnalysisResult:
    metrics: dict
    original: np.ndarray
    preview: np.ndarray
    raw_mask: np.ndarray
    clean_mask: np.ndarray
    candidate_mask: np.ndarray
    overlay: np.ndarray
    roi_colors: dict
    candidate_colors: dict
    hue_hist: np.ndarray
    warnings: list


def analyze_wound(rgb, sensitivity=55, min_area=300, kernel_size=5, roi=None):
    start = perf_counter()
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError('8비트 RGB 이미지가 필요합니다.')
    if not 0 <= sensitivity <= 100 or min_area < 0 or kernel_size not in (1, 3, 5, 7, 9):
        raise ValueError('분석 설정이 올바르지 않습니다.')
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = (0, 0, w, h) if roi is None else map(int, roi)
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x1 >= x2 or y1 >= y2:
        raise ValueError('ROI의 가로·세로 범위를 1픽셀 이상 선택해 주세요.')
    hsv = cv2.cvtColor(rgb[y1:y2, x1:x2], cv2.COLOR_RGB2HSV)
    hue, sat, val = cv2.split(hsv)
    span = round(3 + .12*sensitivity)
    sat_min = round(110 - .8*sensitivity)
    raw = (((hue <= span) | (hue >= 180-span)) & (sat >= sat_min) & (val >= 20)).astype(np.uint8)*255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    clean = cv2.morphologyEx(cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) >= min_area and cv2.contourArea(c) > 0]
    candidate = np.zeros_like(clean)
    overlay, preview = rgb.copy(), rgb.copy()
    cv2.rectangle(preview, (x1, y1), (x2-1, y2-1), (30, 190, 230), 2)
    warnings = []
    area = perimeter = compactness = 0.
    mean_hsv = [None, None, None]
    hist = np.zeros(180, int)
    if valid:
        contour = max(valid, key=cv2.contourArea)
        area, perimeter = float(cv2.contourArea(contour)), float(cv2.arcLength(contour, True))
        compactness = min(1., 4*np.pi*area/perimeter**2) if perimeter else 0.
        cv2.drawContours(candidate, [contour], -1, 255, -1)
        selected = candidate > 0
        mean_hsv = hsv[selected].mean(axis=0).tolist()
        hist = np.bincount(hsv[:,:,0][selected], minlength=180)
        section = overlay[y1:y2, x1:x2]
        section[selected] = (section[selected]*.65 + np.array([30, 210, 190])*.35).astype(np.uint8)
        cv2.drawContours(section, [contour], -1, (0, 255, 195), 2)
        bx, by, bw, bh = cv2.boundingRect(contour)
        if bx == 0 or by == 0 or bx+bw >= x2-x1 or by+bh >= y2-y1:
            warnings.append('후보가 분석 범위 경계에 닿습니다. ROI가 상처를 잘랐는지 확인하세요.')
        if len(valid) > 1:
            warnings.append(f'유효 후보 {len(valid)}개 중 가장 큰 1개만 측정했습니다.')
    else:
        warnings.append('조건을 만족하는 후보가 없습니다. ROI와 색상 민감도, 최소 면적을 조정하세요.')
    metrics = dict(detected=bool(valid), area_px2=area, perimeter_px=perimeter,
        image_ratio_pct=100*area/(w*h), roi_ratio_pct=100*area/((x2-x1)*(y2-y1)),
        compactness=compactness, candidate_count=len(valid), mean_h=mean_hsv[0], mean_s=mean_hsv[1], mean_v=mean_hsv[2],
        width=w, height=h, roi=[x1,y1,x2,y2], sensitivity=sensitivity, min_area=min_area, kernel_size=kernel_size,
        seconds=perf_counter()-start)
    return AnalysisResult(metrics, rgb, preview, raw, clean, candidate, overlay,
        color_ratios(hsv), color_ratios(hsv, candidate), hist, warnings)


def compare_results(previous, current):
    if not previous.metrics['detected'] or not current.metrics['detected']:
        return None
    p = previous.metrics['area_px2']
    return 100*(current.metrics['area_px2']-p)/p if p > 0 else None


def sample_image(kind='현재'):
    """Synthetic illustrations, never clinical images or validation data."""
    im = np.full((540, 800, 3), (203, 163, 137), np.uint8)
    axes = (120, 82) if kind == '이전' else (85, 58)
    if kind == '작은 후보': axes = (25, 18)
    color = (105, 24, 35) if kind == '어두운 후보' else (168, 42, 55)
    cv2.ellipse(im, (400,270), axes, -12, 0, 360, color, -1)
    cv2.ellipse(im, (380,260), (max(8,axes[0]//3), max(5,axes[1]//3)), 0, 0, 360, (201,68,78), -1)
    if kind == '배경 오검출': cv2.rectangle(im, (10,50), (220,450), (170,25,40), -1)
    if kind == '그림자': im[:, :350] = (im[:, :350]*.55).astype(np.uint8)
    return im

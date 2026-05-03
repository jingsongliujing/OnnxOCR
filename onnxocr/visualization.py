from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_FONT_PATH = str(MODULE_DIR / "fonts" / "simfang.ttf")


def _load_font(font_path: str, size: int):
    try:
        return ImageFont.truetype(font_path, size, encoding="utf-8")
    except Exception:
        return ImageFont.load_default()


def _put_text(
    img: np.ndarray,
    text: str,
    org,
    color=(255, 0, 0),
    font_size: int = 24,
    font_path: str = DEFAULT_FONT_PATH,
) -> np.ndarray:
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    draw.text(org, text, fill=color, font=_load_font(font_path, font_size))
    return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)


def _put_label(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    font_size: int = 20,
    font_path: str = DEFAULT_FONT_PATH,
) -> np.ndarray:
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = _load_font(font_path, font_size)
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]

    img_w, img_h = pil_img.size
    x = max(0, min(x, max(0, img_w - text_w - 8)))
    y = max(0, min(y, max(0, img_h - text_h - 8)))
    draw.rectangle((x, y, x + text_w + 8, y + text_h + 8), fill=(255, 255, 255))
    draw.rectangle((x, y, x + text_w + 8, y + text_h + 8), outline=(255, 0, 0), width=1)
    draw.text((x + 4, y + 3), text, fill=(255, 0, 0), font=font)
    return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)


def draw_plate_recognition(
    img: np.ndarray,
    results: List[Dict],
    font_path: str = DEFAULT_FONT_PATH,
) -> np.ndarray:
    vis_img = img.copy()
    landmark_colors = [(255, 0, 0), (0, 180, 0), (0, 0, 255), (0, 180, 180)]

    for item in results:
        x1, y1, x2, y2 = [int(v) for v in item.get("axis", [])]
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        for idx, point in enumerate(item.get("landmarks", [])):
            px, py = int(point[0]), int(point[1])
            cv2.circle(vis_img, (px, py), 4, landmark_colors[idx % len(landmark_colors)], -1)

        plate = item.get("plate", "")
        score = item.get("score", 0)
        plate_type = item.get("type", "")
        label = f"{plate} {score:.2f} {plate_type}".strip()
        text_y = y1 - 30 if y1 >= 30 else y2 + 4
        vis_img = _put_label(vis_img, label, x1, text_y, 20, font_path)

    return vis_img


def draw_table_recognition(
    img: np.ndarray,
    result: Dict,
    show_logic: bool = True,
) -> np.ndarray:
    vis_img = img.copy()
    cell_bboxes = np.array(result.get("cell_bboxes", []), dtype=np.float32)
    logic_points = result.get("logic_points", [])

    if cell_bboxes.size == 0:
        return vis_img

    for idx, box in enumerate(cell_bboxes):
        if box.shape[0] == 4:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            text_x, text_y = x1 + 3, y1 + 16
        elif box.shape[0] == 8:
            points = box.reshape(4, 2).astype(int)
            cv2.polylines(vis_img, [points], True, (255, 0, 0), 2)
            text_x, text_y = int(points[:, 0].min()) + 3, int(points[:, 1].min()) + 16
        else:
            continue

        if show_logic and idx < len(logic_points):
            logic = logic_points[idx]
            label = f"r{logic[0]}-{logic[1]} c{logic[2]}-{logic[3]}"
            cv2.putText(
                vis_img,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

    return vis_img


def draw_layout_analysis(
    img: np.ndarray,
    result: Dict,
    font_path: str = DEFAULT_FONT_PATH,
) -> np.ndarray:
    vis_img = img.copy()
    boxes = result.get("boxes", [])
    class_names = result.get("class_names", [])
    scores = result.get("scores", [])

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        color = (
            int(37 + idx * 53) % 255,
            int(128 + idx * 97) % 255,
            int(214 + idx * 31) % 255,
        )
        overlay = vis_img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        vis_img = cv2.addWeighted(overlay, 0.18, vis_img, 0.82, 0)
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)

        label = class_names[idx] if idx < len(class_names) else "layout"
        score = scores[idx] if idx < len(scores) else 0
        vis_img = _put_label(vis_img, f"{label} {score:.2f}", x1, max(0, y1 - 28), 18, font_path)

    return vis_img


def save_plate_visualization(
    img: np.ndarray,
    results: List[Dict],
    name: str,
    font_path: str = DEFAULT_FONT_PATH,
) -> None:
    cv2.imwrite(name, draw_plate_recognition(img, results, font_path=font_path))


def save_table_visualization(
    img: np.ndarray,
    result: Dict,
    name: str,
    show_logic: bool = False,
) -> None:
    cv2.imwrite(name, draw_table_recognition(img, result, show_logic=show_logic))


def save_layout_visualization(
    img: np.ndarray,
    result: Dict,
    name: str,
    font_path: str = DEFAULT_FONT_PATH,
) -> None:
    cv2.imwrite(name, draw_layout_analysis(img, result, font_path=font_path))


def image_to_base64(img: np.ndarray, ext: str = ".jpg") -> Optional[str]:
    import base64

    ok, buffer = cv2.imencode(ext, img)
    if not ok:
        return None
    return base64.b64encode(buffer).decode("ascii")

import os
import time
import gc
import uuid

import cv2
from PIL import Image
import importlib
import numpy as np
from loguru import logger

try:
    import torch
    import torch_npu
except ImportError:
    pass

def import_package(name, package=None):
    try:
        module = importlib.import_module(name, package=package)
        return module
    except ModuleNotFoundError:
        return None

def check_openvino():
    """
    检查当前环境是否支持 OpenVINO
    """
    try:
        try:
            from openvino import Core
        except ImportError:
            from openvino.runtime import Core  # 兼容旧版本
        core = Core()
        devices = core.available_devices
        return bool(devices)
    except Exception as e:
        print(f"OpenVINO 可用性检查出错: {e}")
        return False

def crop_img(input_res, input_img: np.ndarray, crop_paste_x=0, crop_paste_y=0, layout_shape_mode="auto"):

    crop_xmin, crop_ymin = int(input_res['poly'][0]), int(input_res['poly'][1])
    crop_xmax, crop_ymax = int(input_res['poly'][4]), int(input_res['poly'][5])

    # Calculate new dimensions
    crop_new_width = crop_xmax - crop_xmin + crop_paste_x * 2
    crop_new_height = crop_ymax - crop_ymin + crop_paste_y * 2

    # Create a white background array
    return_image = np.ones((crop_new_height, crop_new_width, 3), dtype=np.uint8) * 255

    # Crop the original image using numpy slicing
    cropped_img = input_img[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

    polygon = input_res.get("polygon_points")
    if layout_shape_mode != "rect" and polygon:
        polygon = np.array(polygon, dtype=np.int32)
        if polygon.ndim == 1:
            polygon = polygon.reshape((-1, 2))
        polygon = polygon.reshape((-1, 1, 2))
        polygon = polygon - np.array([crop_xmin, crop_ymin])
        mask = np.zeros(cropped_img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 1)
        mask = mask.astype(bool)
        cropped_img = cropped_img.copy()
        cropped_img[~mask] = 255

    # Paste the cropped image onto the white background
    return_image[crop_paste_y:crop_paste_y + (crop_ymax - crop_ymin),
    crop_paste_x:crop_paste_x + (crop_xmax - crop_xmin)] = cropped_img

    return_list = [crop_paste_x, crop_paste_y, crop_xmin, crop_ymin, crop_xmax, crop_ymax, crop_new_width,
                   crop_new_height]
    return return_image, return_list

def get_coords_and_area(block_with_poly):
    """Extract coordinates and area from a table."""
    xmin, ymin = int(block_with_poly['poly'][0]), int(block_with_poly['poly'][1])
    xmax, ymax = int(block_with_poly['poly'][4]), int(block_with_poly['poly'][5])
    area = (xmax - xmin) * (ymax - ymin)
    return xmin, ymin, xmax, ymax, area


def calculate_intersection(box1, box2):
    """Calculate intersection coordinates between two boxes."""
    intersection_xmin = max(box1[0], box2[0])
    intersection_ymin = max(box1[1], box2[1])
    intersection_xmax = min(box1[2], box2[2])
    intersection_ymax = min(box1[3], box2[3])

    # Check if intersection is valid
    if intersection_xmax <= intersection_xmin or intersection_ymax <= intersection_ymin:
        return None

    return intersection_xmin, intersection_ymin, intersection_xmax, intersection_ymax


def is_inside(small_box, big_box, overlap_threshold=0.8):
    """Check if small_box is inside big_box by at least overlap_threshold."""
    intersection = calculate_intersection(small_box[:4], big_box[:4])

    if not intersection:
        return False

    intersection_xmin, intersection_ymin, intersection_xmax, intersection_ymax = intersection
    intersection_area = (intersection_xmax - intersection_xmin) * (intersection_ymax - intersection_ymin)

    # Check if overlap exceeds threshold
    return intersection_area >= overlap_threshold * small_box[4]


def get_res_list_from_layout_res(layout_res, np_img, overlap_threshold=0.8):
    """Extract OCR, table and other regions from layout results."""
    ocr_res_list = []
    table_res_list = []
    single_page_mfdetrec_res = []
    image_res_list = []

    # Categorize regions
    for i, res in enumerate(layout_res):
        category_id = int(res['category_id'])
        if category_id in [3]:  # Image regions
            image_res_list.append(res)
        if category_id in [8, 13, 14]:  # Formula regions
            res['bbox'] = [int(res['poly'][0]), int(res['poly'][1]), int(res['poly'][4]), int(res['poly'][5])]
            single_page_mfdetrec_res.append(res)
        # elif category_id in [0, 2, 4, 6, 7, 3]:  # OCR regions # 相信版面结果，图片就是图片，不再尝试转为文本块
        elif category_id in [0, 1, 2, 4, 6, 7]:  # OCR regions
            ocr_res_list.append(res)
        elif category_id == 5:  # Table regions
            table_res_list.append(res)

    # 找出所有在表格内的图片框
    for img_idx, img_box in enumerate(image_res_list):
        for tbl_idx, tbl_box in enumerate(table_res_list):
            if is_inside(get_coords_and_area(img_box), get_coords_and_area(tbl_box), overlap_threshold):
                if 'layout_image_list' not in tbl_box:
                    tbl_box['layout_image_list'] = []
                fill_image_dict = {
                    "uuid": str(uuid.uuid4()),
                    "poly": img_box['poly'],
                    'pil_image': Image.fromarray(crop_img(img_box, np_img)[0]),
                }
                tbl_box['layout_image_list'].append(fill_image_dict)

    return ocr_res_list, table_res_list, single_page_mfdetrec_res


def clean_memory(device='cuda'):
    if device == 'cuda':
        torch_ = import_package("torch")
        if torch_ and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    elif str(device).startswith("npu"):
        torch_npu_ = import_package("torch_npu")
        if torch_npu_ and torch_npu.npu.is_available():
            torch_npu.npu.empty_cache()
    elif str(device).startswith("mps"):
        torch_ = import_package("torch")
        if torch_:
            torch.mps.empty_cache()
    gc.collect()

def clean_vram(device, vram_threshold=8):
    vram_threshold = int(os.getenv('MINERU_VRAM_THRESHOLD', vram_threshold))
    total_memory = get_vram(device)
    if total_memory is not None:
        total_memory = int(os.getenv('MINERU_VIRTUAL_VRAM_SIZE', round(total_memory)))
    if total_memory and total_memory <= vram_threshold:
        gc_start = time.time()
        clean_memory(device)
        gc_time = round(time.time() - gc_start, 2)
        logger.info(f"gc time: {gc_time}")


def get_vram(device):
    torch_ = import_package("torch")
    if torch_ and torch.cuda.is_available() and str(device).startswith("cuda"):
        total_memory = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)  # 将字节转换为 GB
        return total_memory
    elif str(device).startswith("npu"):
        torch_npu_ = import_package("torch_npu")
        if torch_npu_ and torch_npu.npu.is_available():
            total_memory = torch_npu.npu.get_device_properties(device).total_memory / (1024 ** 3)  # 转为 GB
            return total_memory
    else:
        return None

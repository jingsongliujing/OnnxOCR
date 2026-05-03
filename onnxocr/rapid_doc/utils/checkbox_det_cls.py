import cv2
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


def detect_checkboxes(image, line_min_width=15, line_max_width=15):
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转为灰度图
    th1, img_bin = cv2.threshold(gray_scale, 150, 255, cv2.THRESH_BINARY)
    img_bin = ~img_bin  # 取反，黑白互换

    # 定义水平和垂直方向的核
    kernal_h_min = np.ones((1, line_min_width), np.uint8)
    kernal_v_min = np.ones((line_min_width, 1), np.uint8)
    kernal_h_max = np.ones((1, line_max_width), np.uint8)
    kernal_v_max = np.ones((line_max_width, 1), np.uint8)

    # 形态学操作提取线条
    img_bin_h_min = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_h_min)
    img_bin_v_min = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_v_min)
    img_bin_h_max = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernal_h_max)
    img_bin_v_max = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernal_v_max)

    img_bin_final = (img_bin_h_min & img_bin_h_max) | (img_bin_v_min & img_bin_v_max)

    final_kernel = np.ones((3, 3), np.uint8)
    img_bin_final = cv2.dilate(img_bin_final, final_kernel, iterations=1)

    # 连通区域分析
    _, labels, stats, _ = cv2.connectedComponentsWithStats(~img_bin_final, connectivity=8, ltype=cv2.CV_32S)

    return stats, labels, img_bin_final


def classify_checkboxes(image, stats, img_bin_final, min_width=12, max_width=50, tick_threshold=0.2, debug_show=False):
    checkboxes = []
    h_img, w_img = img_bin_final.shape
    for stat in stats[2:]:
        x, y, w, h, area = stat
        aspect_ratio = w / h
        if min_width <= w <= max_width and min_width <= h <= max_width and 0.9 <= aspect_ratio <= 1.1:
            # print(x, y, w, h, aspect_ratio)
            # 自适应扩展比例
            expand_left, expand_right = 0.1, 0.1
            expand_top, expand_bottom = 0.1, 0.3  # 向下多扩展

            new_x = max(int(x - w * expand_left), 0)
            new_y = max(int(y - h * expand_top), 0)
            new_w = min(int(w * (1 + expand_left + expand_right)), w_img - new_x)
            new_h = min(int(h * (1 + expand_top + expand_bottom)), h_img - new_y)

            roi = img_bin_final[new_y:new_y + new_h, new_x:new_x + new_w].copy()
            # 找出轮廓
            contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # 将轮廓坐标转换为全图坐标
            contours_global = [contour + np.array([[new_x, new_y]]) for contour in contours]

            all_ys = [pt[0][1] for contour in contours_global for pt in contour]
            # 统计出现次数
            y_counts = Counter(all_ys)
            # 按出现次数排序，出现次数少的通常是噪声或左右边线
            y_counts_sorted = sorted(y_counts.items(), key=lambda x: (-x[1], x[0]))
            # 取出现次数最多的几个 y 值（比如前2个）
            top_ys = [y for y, count in y_counts_sorted[:2]]
            # 最大的 y 值一般就是下边框下边
            bottom_y = max(top_ys)

            y_max = max(all_ys)  # 底部

            diff = y_max - bottom_y

            if diff >= 0.9:  # 高度差值
                continue  # 丢掉这个区域

            roi = image[y:y + h, x:x + w]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)

            binary_roi = cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY_INV, 11, 2)
            kernel = np.ones((3, 3), np.uint8)
            binary_roi = cv2.morphologyEx(binary_roi, cv2.MORPH_OPEN, kernel)

            non_white_pixels = cv2.countNonZero(binary_roi)
            total_pixels = w * h
            tick_percentage = non_white_pixels / total_pixels

            # ☐ 未选框、 ☑ 已选框、 ☒ 带叉的选框 （暂未支持）
            if tick_percentage > tick_threshold:
                checkboxes.append((int(x), int(y), int(w), int(h), 'Ticked', '☑'))
            else:
                checkboxes.append((int(x), int(y), int(w), int(h), 'Unticked', '☐'))
    return checkboxes


def process_image(image_path, plt_show=False, debug_show=False):
    image = cv2.imread(image_path)
    if image is not None:
        stats, labels, img_bin_final = detect_checkboxes(image)

        if debug_show:
            cv2.imshow("Morphology Combined", img_bin_final)
            cv2.waitKey(0)

        checkboxes = classify_checkboxes(image, stats, img_bin_final, debug_show=debug_show)
        if plt_show:
            import matplotlib
            matplotlib.use("TkAgg")

            for checkbox in checkboxes:
                x, y, w, h, label, text = checkbox
                if 'Ticked' == label:
                    checkboxes.append((int(x), int(y), int(w), int(h), label, text))
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                else:
                    checkboxes.append((int(x), int(y), int(w), int(h), label, text))
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
        return checkboxes
    else:
        print(f"无法打开或找到图片: {image_path}")
        return None


def checkbox_predict(image: np.ndarray):
    # 复选框检测
    stats, labels, img_bin_final = detect_checkboxes(image)
    checkbox_results = classify_checkboxes(image, stats, img_bin_final)

    checkbox_res = []
    for checkbox in checkbox_results:
        x, y, w, h, label, text = checkbox
        bbox = [x, y, x + w, y + h]
        checkbox_res.append({
            "bbox": bbox,
            "label": label,
            "text": text
        })
    return checkbox_res

if __name__ == "__main__":
    # image_path = '0a471d47-e428-4e5a-b849-a8aedfe399a0.png' #20 # 0.2702702702702703
    # image_path = 'e3c88dc773959dd02b261c59d5d0c9b6.png'   #35-39 # 0.24647887323943662

    # image_path = 'ce605ac0-00d2-4e5e-8d22-ca646d847ffa.png' # 21  0.275
    # image_path = 'aaaa.png'
    image_path = '../../tests/checkbox_test.png'

    results = process_image(image_path, plt_show=True, debug_show=True)
    if results:
        print(f"{image_path}: {results}")
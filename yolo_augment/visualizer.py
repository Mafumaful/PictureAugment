"""YOLO数据可视化模块 - 使用OpenCV显示图像和标注框"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple

from .annotation import BBox, parse_yolo_label


# 颜色列表 (BGR格式)
COLORS = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0),
    (0, 255, 255), (255, 0, 255), (255, 255, 0),
    (0, 0, 128), (0, 128, 0), (128, 0, 0),
    (0, 128, 128), (128, 0, 128), (128, 128, 0),
]


class YOLOVisualizer:
    """YOLO数据可视化器"""

    def __init__(self, images_dir: str, labels_dir: str):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.current_idx = 0
        self.image_files = self._get_image_files()
        self.window_name = "YOLO Visualizer"

    def _get_image_files(self) -> List[Path]:
        """获取所有图像文件"""
        exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        files = []
        for ext in exts:
            files.extend(self.images_dir.glob(ext))
        return sorted(files)

    def _load_image(self, img_path: Path) -> Tuple[np.ndarray, List[BBox]]:
        """加载图像和标注"""
        img = cv2.imread(str(img_path))
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        bboxes = parse_yolo_label(label_path)
        return img, bboxes

    def _draw_bboxes(self, img: np.ndarray, bboxes: List[BBox]) -> np.ndarray:
        """在图像上绘制边界框"""
        h, w = img.shape[:2]
        result = img.copy()
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox.to_corners(w, h)
            color = COLORS[bbox.class_id % len(COLORS)]
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            label = f"cls:{bbox.class_id}"
            cv2.putText(result, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return result

    def _draw_info(self, img: np.ndarray, img_path: Path, bboxes: List[BBox]) -> np.ndarray:
        """绘制信息文字"""
        result = img.copy()
        info = f"{img_path.name} [{self.current_idx+1}/{len(self.image_files)}] boxes:{len(bboxes)}"
        cv2.putText(result, info, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        help_text = "Left/Right: switch | ESC/Q: quit"
        cv2.putText(result, help_text, (10, img.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        return result

    def run(self):
        """运行可视化器"""
        if not self.image_files:
            print("未找到图像文件")
            return

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)

        print(f"找到 {len(self.image_files)} 张图像")
        print("按 ← → 切换图片, ESC/Q 退出")

        while True:
            img_path = self.image_files[self.current_idx]
            img, bboxes = self._load_image(img_path)
            img = self._draw_bboxes(img, bboxes)
            img = self._draw_info(img, img_path, bboxes)

            cv2.imshow(self.window_name, img)
            key = cv2.waitKey(30) & 0xFF

            # ESC或Q退出
            if key == 27 or key == ord('q'):
                break
            # 左箭头
            elif key == 81 or key == 2:
                self.current_idx = (self.current_idx - 1) % len(self.image_files)
            # 右箭头
            elif key == 83 or key == 3:
                self.current_idx = (self.current_idx + 1) % len(self.image_files)

        cv2.destroyAllWindows()

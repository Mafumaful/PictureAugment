"""YOLO标注解析和保存模块"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import numpy as np


@dataclass
class BBox:
    """YOLO边界框"""
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def to_corners(self, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        """转换为像素坐标 (x1, y1, x2, y2)"""
        x1 = int((self.x_center - self.width / 2) * img_w)
        y1 = int((self.y_center - self.height / 2) * img_h)
        x2 = int((self.x_center + self.width / 2) * img_w)
        y2 = int((self.y_center + self.height / 2) * img_h)
        return x1, y1, x2, y2

    @classmethod
    def from_corners(cls, class_id: int, x1: int, y1: int, x2: int, y2: int,
                     img_w: int, img_h: int) -> "BBox":
        """从像素坐标创建"""
        x_center = ((x1 + x2) / 2) / img_w
        y_center = ((y1 + y2) / 2) / img_h
        width = (x2 - x1) / img_w
        height = (y2 - y1) / img_h
        return cls(class_id, x_center, y_center, width, height)

    def to_line(self) -> str:
        """转换为YOLO格式行"""
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"


def parse_yolo_label(label_path: Path) -> List[BBox]:
    """解析YOLO标注文件"""
    bboxes = []
    if not label_path.exists():
        return bboxes

    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                bbox = BBox(
                    class_id=int(parts[0]),
                    x_center=float(parts[1]),
                    y_center=float(parts[2]),
                    width=float(parts[3]),
                    height=float(parts[4])
                )
                bboxes.append(bbox)
    return bboxes


def save_yolo_label(bboxes: List[BBox], label_path: Path) -> None:
    """保存YOLO标注文件"""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, 'w') as f:
        for bbox in bboxes:
            f.write(bbox.to_line() + '\n')

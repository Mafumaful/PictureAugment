"""数据增强器模块"""

import cv2
from pathlib import Path
from typing import List
from tqdm import tqdm

from .annotation import parse_yolo_label, save_yolo_label
from .transforms import build_transforms


class YOLOAugmenter:
    """YOLO数据增强器"""

    def __init__(self, config: dict):
        self.config = config
        self.input_images_root = Path(config['input_images_root'])
        self.input_labels_root = Path(config['input_labels_root'])
        self.output_images_root = Path(config['output_images_root'])
        self.output_labels_root = Path(config['output_labels_root'])
        self.datasets = config.get('datasets', ['train'])
        self.num_augments = config.get('num_augments', 1)
        self.transforms = build_transforms(config.get('transforms', {}))

    def _get_image_files(self, images_dir: Path) -> List[Path]:
        """获取目录下所有图像文件"""
        exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        files = []
        for ext in exts:
            files.extend(images_dir.glob(ext))
        return sorted(files)

    def run(self) -> None:
        """执行增强"""
        for dataset in self.datasets:
            self._process_dataset(dataset)

    def _process_dataset(self, dataset: str) -> None:
        """处理单个数据集"""
        images_dir = self.input_images_root / dataset
        labels_dir = self.input_labels_root / dataset
        out_images_dir = self.output_images_root / dataset
        out_labels_dir = self.output_labels_root / dataset

        out_images_dir.mkdir(parents=True, exist_ok=True)
        out_labels_dir.mkdir(parents=True, exist_ok=True)

        images = self._get_image_files(images_dir)
        if not images:
            print(f"[{dataset}] 未找到图像")
            return

        print(f"[{dataset}] 找到 {len(images)} 张图像")

        total = len(images) * self.num_augments
        with tqdm(total=total, desc=f"[{dataset}]") as pbar:
            for img_path in images:
                self._process_image(img_path, labels_dir,
                                    out_images_dir, out_labels_dir, pbar)

    def _process_image(self, img_path: Path, labels_dir: Path,
                       out_images_dir: Path, out_labels_dir: Path, pbar) -> None:
        """处理单张图像"""
        label_path = labels_dir / f"{img_path.stem}.txt"
        image = cv2.imread(str(img_path))
        if image is None:
            pbar.update(self.num_augments)
            return

        bboxes = parse_yolo_label(label_path)

        for i in range(self.num_augments):
            aug_img, aug_bboxes = self.transforms(image.copy(), bboxes.copy())
            suffix = f"_aug{i}"
            out_img = out_images_dir / f"{img_path.stem}{suffix}{img_path.suffix}"
            out_label = out_labels_dir / f"{img_path.stem}{suffix}.txt"

            cv2.imwrite(str(out_img), aug_img)
            save_yolo_label(aug_bboxes, out_label)
            pbar.update(1)

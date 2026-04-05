"""配置文件解析模块"""

from pathlib import Path
from typing import Dict, Any
import yaml


def load_config(config_path: Path) -> Dict[str, Any]:
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """验证配置文件"""
    required = ['input_images_root', 'input_labels_root',
                'output_images_root', 'output_labels_root']
    for key in required:
        if key not in config:
            raise ValueError(f"配置缺少必需字段: {key}")

    transforms = config.get('transforms', {})
    valid_transforms = ['rotate', 'mask', 'distort', 'gaussian_noise',
                        'hsv_adjust', 'grayscale', 'brightness_boost']
    for name in transforms:
        if name not in valid_transforms:
            raise ValueError(f"未知变换类型: {name}")

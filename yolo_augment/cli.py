"""命令行接口"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='YOLO数据增强工具')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # augment命令
    aug_parser = subparsers.add_parser('augment', help='数据增强')
    aug_parser.add_argument('-c', '--config', type=str, required=True,
                            help='配置文件路径')

    # view命令
    view_parser = subparsers.add_parser('view', help='可视化查看')
    view_parser.add_argument('-d', '--data', type=str, default='./data',
                             help='数据根目录 (默认: ./data)')
    view_parser.add_argument('-s', '--split', type=str, default='train',
                             help='数据集: train/val/test (默认: train)')

    args = parser.parse_args()

    if args.command == 'augment':
        from .config import load_config, validate_config
        from .augmenter import YOLOAugmenter

        config_path = Path(args.config)
        if not config_path.exists():
            print(f"配置文件不存在: {config_path}")
            return 1

        config = load_config(config_path)
        validate_config(config)
        augmenter = YOLOAugmenter(config)
        augmenter.run()
        print("增强完成!")
        return 0

    elif args.command == 'view':
        from .visualizer import YOLOVisualizer

        data_root = Path(args.data)
        split = args.split
        images_dir = data_root / 'images' / split
        labels_dir = data_root / 'labels' / split

        if not images_dir.exists():
            print(f"图像目录不存在: {images_dir}")
            return 1

        visualizer = YOLOVisualizer(str(images_dir), str(labels_dir))
        visualizer.run()
        return 0

    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    exit(main())

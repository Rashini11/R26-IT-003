import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description='Run YOLOv8 inference on images or a folder.')
    parser.add_argument('--weights', type=str, default='runs/detect/train-3/weights/best.pt',
                        help='Path to trained model weights')
    parser.add_argument('--source', type=str, default='data/valid/images',
                        help='Image file or folder to run inference on')
    parser.add_argument('--output', type=str, default='runs/inference',
                        help='Directory to save inference results')
    return parser.parse_args()


def main():
    args = parse_args()
    weights_path = Path(args.weights)
    source_path = Path(args.source)
    output_path = Path(args.output)
    weights_path = weights_path.resolve()
    source_path = source_path.resolve()
    output_path = output_path.resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    if not weights_path.exists():
        raise FileNotFoundError(f'Trained weights not found: {weights_path}')
    if not source_path.exists():
        raise FileNotFoundError(f'Source image or folder not found: {source_path}')

    model = YOLO(str(weights_path))
    results = model(str(source_path), save=True, project=str(output_path), name='results', exist_ok=True)

    print('Inference complete.')
    print(f'Saved results to: {output_path / "results"}')
    print(f'Total images processed: {len(results)}')


if __name__ == '__main__':
    main()

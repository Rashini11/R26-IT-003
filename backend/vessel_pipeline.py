#!/usr/bin/env python3
"""
Sequential Decision Tree Pipeline:
1) Detect sl_flag -> Local vs Foreign per vessel
2) Vessel type: boat or ship
3) Final labels: Local Boat / Local Ship / Foreign Boat / Foreign Ship

Usage:
python vessel_pipeline.py --weights path/to/weights.pt --source path/to/image.jpg --out out.jpg
"""
import argparse
import cv2
import numpy as np
from ultralytics import YOLO

VESSEL_CLASS_BOAT = 0
VESSEL_CLASS_SHIP = 1
VESSEL_CLASS_SL_FLAG = 2

# Optional classifier dependencies
try:
    import torch
    import torchvision
    from torchvision import transforms
    from PIL import Image
except Exception:
    torch = None
    transforms = None
    Image = None

def area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)

def intersection_area(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    return w * h

def flag_within_vessel(vessel_box, flag_box, min_flag_overlap=0.25):
    """
    Determine whether the flag lies inside / significantly overlaps the vessel box.
    We consider the flag to be 'within' if intersection_area / flag_area >= min_flag_overlap.
    """
    inter = intersection_area(vessel_box, flag_box)
    if inter <= 0:
        return False
    f_area = area(flag_box)
    if f_area <= 0:
        return False
    return (inter / f_area) >= min_flag_overlap

def draw_box(img, box, label=None, color=(0,255,0), thickness=2):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        t = 2
        # background
        (w, h), _ = cv2.getTextSize(label, font, scale, t)
        cv2.rectangle(img, (x1, y1 - h - 6), (x1 + w + 6, y1), color, -1)
        cv2.putText(img, label, (x1+3, y1 - 6), font, scale, (255,255,255), t, cv2.LINE_AA)

def load_classifier(path=None, arch="resnet50", num_classes=2, device=None):
    if path is None:
        return None
    if torch is None:
        raise RuntimeError("PyTorch is required to load a classifier. Install torch and torchvision.")
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    # create model
    model = torchvision.models.__dict__.get(arch)(pretrained=False)
    # adapt final layer
    if hasattr(model, "fc"):
        in_f = model.fc.in_features
        model.fc = torch.nn.Linear(in_f, num_classes)
    elif hasattr(model, "classifier"):
        # for some models like mobilenet/vgg
        if isinstance(model.classifier, torch.nn.Sequential):
            last = list(model.classifier.children())[-1]
            if hasattr(last, "in_features"):
                in_f = last.in_features
                model.classifier = torch.nn.Sequential(torch.nn.Linear(in_f, num_classes))
    model.to(device)
    # load weights
    sd = torch.load(path, map_location=device)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    try:
        model.load_state_dict(sd)
    except Exception:
        # try direct load (if saved with model.state_dict())
        model.load_state_dict(sd)
    model.eval()
    return model


def preprocess_crop_for_classifier(crop_bgr, input_size=224):
    if Image is None or transforms is None:
        raise RuntimeError("PIL / torchvision.transforms required for classifier preprocessing")
    # convert BGR (OpenCV) to RGB PIL image
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(crop_rgb)
    tf = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return tf(pil).unsqueeze(0)


def run_pipeline(weights, source, outpath=None, flag_overlap_thresh=0.25, show=False,
                 classifier_path=None, classifier_arch="resnet50", classifier_input=224,
                 save_crops=None):
    # load detector
    model = YOLO(weights)

    # load classifier if provided
    clf = None
    device = None
    if classifier_path is not None:
        if torch is None:
            raise RuntimeError("Classifier requested but PyTorch is not installed.")
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        clf = load_classifier(classifier_path, arch=classifier_arch, num_classes=2, device=device)

    # Read image (or first frame from video)
    cap = None
    img = None
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Unable to read frame from {source}")
    img = frame.copy()

    # Run inference
    results = model(img, conf=0.50, iou=0.45, verbose=False)[0]
    boxes = []
    if hasattr(results, "boxes") and len(results.boxes) > 0:
        xyxy = results.boxes.xyxy.cpu().numpy()  # N x 4
        cls = results.boxes.cls.cpu().numpy().astype(int)  # N
        conf = results.boxes.conf.cpu().numpy()  # N
        for b, c, confv in zip(xyxy, cls, conf):
            boxes.append({"xyxy": b.tolist(), "class": int(c), "conf": float(confv)})
    else:
        # No detections
        print("No detections found.")
        if outpath:
            cv2.imwrite(outpath, img)
        if show:
            cv2.imshow("out", img); cv2.waitKey(0); cv2.destroyAllWindows()
        return img

    # Separate detections
    vessels = []  # boat or ship
    flags = []    # sl_flag
    for det in boxes:
        if det["class"] in (VESSEL_CLASS_BOAT, VESSEL_CLASS_SHIP):
            vessels.append(det)
        elif det["class"] == VESSEL_CLASS_SL_FLAG:
            flags.append(det)

    # Keep one highest-confidence vessel and one highest-confidence flag.
    vessels = sorted(vessels, key=lambda detection: detection["conf"], reverse=True)[:1]
    flags = sorted(flags, key=lambda detection: detection["conf"], reverse=True)[:1]

    # For each vessel, check flags and assign final label
    h_img, w_img = img.shape[:2]
    for idx, v in enumerate(vessels):
        vbox = v["xyxy"]
        is_local = False
        matched_flags = []
        for f in flags:
            fbox = f["xyxy"]
            if flag_within_vessel(vbox, fbox, min_flag_overlap=flag_overlap_thresh):
                is_local = True
                matched_flags.append(f)

        # If no flag inside vessel and classifier is available, crop and run classifier
        clf_result = None
        if not is_local and clf is not None:
            # crop vessel with small padding
            x1, y1, x2, y2 = map(int, vbox)
            pad_w = int((x2 - x1) * 0.15)
            pad_h = int((y2 - y1) * 0.15)
            cx1 = max(0, x1 - pad_w)
            cy1 = max(0, y1 - pad_h)
            cx2 = min(w_img - 1, x2 + pad_w)
            cy2 = min(h_img - 1, y2 + pad_h)
            crop = img[cy1:cy2, cx1:cx2]
            if save_crops:
                import os
                os.makedirs(save_crops, exist_ok=True)
                cv2.imwrite(os.path.join(save_crops, f"crop_{idx}.jpg"), crop)
            try:
                inp = preprocess_crop_for_classifier(crop, input_size=classifier_input).to(device)
                with torch.no_grad():
                    out = clf(inp)
                    if out.shape[1] == 1:
                        prob = torch.sigmoid(out)[0][0].item()
                        pred = 1 if prob >= 0.5 else 0
                    else:
                        pred = int(torch.argmax(out, dim=1).cpu().numpy()[0])
                # mapping: classifier class 1 -> Local, 0 -> Foreign
                clf_result = True if pred == 1 else False
                if clf_result:
                    is_local = True
            except Exception as e:
                print("Classifier inference failed:", e)

        v_type = "Boat" if v["class"] == VESSEL_CLASS_BOAT else "Ship"
        final_label = ("Local " if is_local else "Foreign ") + v_type
        color = (0,255,0) if is_local else (0,0,255)  # BGR: green for Local, red for Foreign

        # Draw vessel box with final label
        draw_box(img, vbox, label=final_label, color=color, thickness=3)

        # Draw the selected flag in yellow, or the selected unmatched flag in cyan.
        if matched_flags:
            for mf in matched_flags:
                draw_box(img, mf["xyxy"], label="sl_flag", color=(0,215,255), thickness=2)
        else:
            for f in flags:
                draw_box(img, f["xyxy"], label="sl_flag", color=(255,255,0), thickness=1)

    # If there are flags but no vessels, draw flags alone
    if len(vessels) == 0 and len(flags) > 0:
        for f in flags:
            draw_box(img, f["xyxy"], label="sl_flag", color=(255,255,0), thickness=2)

    # Save or show
    if outpath:
        cv2.imwrite(outpath, img)
        print(f"Output saved to {outpath}")
    if show:
        cv2.imshow("result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", "-w", required=True, help="Path to YOLOv8 weights .pt for detector")
    p.add_argument("--source", "-s", required=True, help="Image file path or video file path or camera index (0,1...)")
    p.add_argument("--out", "-o", default=None, help="Output image path to save annotated result")
    p.add_argument("--show", action="store_true", help="Show result in a window")
    p.add_argument("--flag-overlap-thresh", type=float, default=0.25, help="Min fraction of flag area overlapping vessel to consider it 'within'")
    p.add_argument("--classifier", help="Path to PyTorch classifier checkpoint (.pth). Class mapping: 1=Local, 0=Foreign")
    p.add_argument("--classifier-arch", default="resnet50", help="Classifier architecture name from torchvision (default: resnet50)")
    p.add_argument("--classifier-input", type=int, default=224, help="Classifier input size (default: 224)")
    p.add_argument("--save-crops", default=None, help="Optional folder to save cropped vessel images for training/debug")
    args = p.parse_args()

    run_pipeline(args.weights, args.source, outpath=args.out, flag_overlap_thresh=args.flag_overlap_thresh,
                 show=args.show, classifier_path=args.classifier, classifier_arch=args.classifier_arch,
                 classifier_input=args.classifier_input, save_crops=args.save_crops)
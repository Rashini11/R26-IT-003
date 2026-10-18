from pathlib import Path
import json
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]

SOURCE_MODEL = (
    ROOT
    / "ml"
    / "models"
    / "ais_motion_under90_final"
    / "ais_motion_gru_under90_best.pth"
)

CONFIG_PATH = (
    ROOT
    / "ml"
    / "models"
    / "ais_motion_under90_final"
    / "config.json"
)

TRAIN_NPZ = (
    ROOT
    / "ml"
    / "ais_motion"
    / "sequences_5min"
    / "train_motion_sequences.npz"
)

METADATA_PATH = (
    ROOT
    / "ml"
    / "ais_motion"
    / "sequences_5min"
    / "metadata.json"
)

OUTPUT = (
    ROOT
    / "ml"
    / "models"
    / "final"
    / "ais_motion_gru_under90_deploy.pth"
)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

config = json.loads(
    CONFIG_PATH.read_text()
)

metadata = json.loads(
    METADATA_PATH.read_text()
)

train = np.load(TRAIN_NPZ)

position = train[
    "y_position"
].astype(np.float32)

speed = train[
    "y_speed"
].astype(np.float32)

position_mean = position.mean(
    axis=0
)

position_std = position.std(
    axis=0
)

position_std = np.where(
    position_std < 1e-6,
    1.0,
    position_std,
)

speed_mean = float(
    speed.mean()
)

speed_std = float(
    speed.std()
)

if speed_std < 1e-6:
    speed_std = 1.0

state_dict = torch.load(
    SOURCE_MODEL,
    map_location="cpu",
    weights_only=True,
)

checkpoint = {
    "model_type": "gru",
    "model_version":
        "ais_motion_gru_under90_final",

    "input_size": 7,
    "hidden_size":
        int(config["hidden"]),

    "shared_size":
        int(config["shared"]),

    "num_layers": 1,

    "dropout":
        float(config["dropout"]),

    "class_names":
        metadata["class_names"],

    "model_state_dict":
        state_dict,

    "position_mean":
        position_mean.tolist(),

    "position_std":
        position_std.tolist(),

    "speed_mean":
        speed_mean,

    "speed_std":
        speed_std,

    "validation_accuracy_percent":
        85.15,

    "test_accuracy_percent":
        86.76,

    "test_macro_f1":
        0.5521,

    "test_position_mae_metres":
        119.38,

    "test_position_median_error_metres":
        11.43,

    "test_speed_mae_knots":
        0.632,
}

torch.save(
    checkpoint,
    OUTPUT,
)

print("Deployment checkpoint created:")
print(OUTPUT)

print()
print("Hidden size :", checkpoint["hidden_size"])
print("Shared size :", checkpoint["shared_size"])
print("Layers      :", checkpoint["num_layers"])
print("Validation  :", checkpoint["validation_accuracy_percent"])
print("Test        :", checkpoint["test_accuracy_percent"])

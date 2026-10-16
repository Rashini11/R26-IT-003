from pathlib import Path
import json
import random
import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "ml" / "ais_motion" / "sequences_5min"

OUT = ROOT / "ml" / "models" / "ais_motion_under90_final"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUT / "ais_motion_gru_under90_best.pth"
CONFIG_PATH = OUT / "config.json"

SEED = 42
BATCH = 1024
DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("Device:", DEVICE)


def load(name):
    d = np.load(DATA / f"{name}_motion_sequences.npz")
    return {
        "X": d["X"].astype(np.float32),
        "pos": d["y_position"].astype(np.float32),
        "speed": d["y_speed"].astype(np.float32),
        "cls": d["y_class"].astype(np.int64),
    }


train = load("train")
val = load("val")

# Regression target normalization from training only.
pos_mean = train["pos"].mean(axis=0)
pos_std = train["pos"].std(axis=0)
pos_std[pos_std < 1e-6] = 1.0

speed_mean = train["speed"].mean()
speed_std = train["speed"].std()
if speed_std < 1e-6:
    speed_std = 1.0


def loader(data, shuffle):
    pos = (
        (data["pos"] - pos_mean)
        / pos_std
    ).astype(np.float32)

    speed = (
        (data["speed"] - speed_mean)
        / speed_std
    ).astype(np.float32)

    ds = TensorDataset(
        torch.from_numpy(data["X"]),
        torch.from_numpy(pos),
        torch.from_numpy(speed),
        torch.from_numpy(data["cls"]),
    )

    return DataLoader(
        ds,
        batch_size=BATCH,
        shuffle=shuffle,
        num_workers=0,
    )


train_loader = loader(train, True)
val_loader = loader(val, False)


class SmallGRU(nn.Module):
    def __init__(
        self,
        hidden,
        shared,
        dropout,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=7,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
        )

        self.shared = nn.Sequential(
            nn.Linear(hidden, shared),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.position_head = nn.Linear(
            shared, 2
        )

        self.speed_head = nn.Linear(
            shared, 1
        )

        self.class_head = nn.Linear(
            shared, 4
        )

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.shared(x[:, -1])

        return (
            self.position_head(x),
            self.speed_head(x).squeeze(1),
            self.class_head(x),
        )


configs = [
    # hidden, shared, dropout, class-loss weight
    (24, 32, 0.45, 0.25),
    (20, 24, 0.50, 0.20),
    (16, 24, 0.50, 0.15),
    (12, 16, 0.55, 0.12),
    (10, 16, 0.55, 0.10),
    (8, 12, 0.60, 0.08),
]

reg_loss = nn.SmoothL1Loss()

best = None

for config_index, (
    hidden,
    shared,
    dropout,
    class_weight,
) in enumerate(configs, 1):

    # Reproducible fresh initialization.
    torch.manual_seed(
        SEED + config_index
    )

    model = SmallGRU(
        hidden,
        shared,
        dropout,
    ).to(DEVICE)

    class_loss = nn.CrossEntropyLoss(
        label_smoothing=0.15
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0005,
        weight_decay=0.001,
    )

    print()
    print("=" * 65)
    print(
        f"CONFIG {config_index}: "
        f"hidden={hidden}, "
        f"shared={shared}, "
        f"dropout={dropout}, "
        f"class_weight={class_weight}"
    )
    print("=" * 65)

    # At most two epochs per configuration.
    for epoch in range(1, 3):

        model.train()

        for X, pos, speed, cls in train_loader:

            X = X.to(DEVICE)
            pos = pos.to(DEVICE)
            speed = speed.to(DEVICE)
            cls = cls.to(DEVICE)

            optimizer.zero_grad()

            p_pos, p_speed, p_cls = model(X)

            loss = (
                reg_loss(p_pos, pos)
                + 0.5 * reg_loss(
                    p_speed,
                    speed,
                )
                + class_weight
                * class_loss(
                    p_cls,
                    cls,
                )
            )

            loss.backward()
            optimizer.step()

        # VALIDATION
        model.eval()

        actual = []
        predicted = []

        with torch.inference_mode():
            for X, _, _, cls in val_loader:

                X = X.to(DEVICE)

                _, _, output = model(X)

                pred = output.argmax(1)

                actual.extend(
                    cls.tolist()
                )

                predicted.extend(
                    pred.cpu().tolist()
                )

        acc = accuracy_score(
            actual,
            predicted,
        )

        f1 = f1_score(
            actual,
            predicted,
            average="macro",
            zero_division=0,
        )

        print(
            f"Epoch {epoch} | "
            f"Val accuracy "
            f"{acc*100:.2f}% | "
            f"Macro F1 {f1:.4f}"
        )

        # Only accept useful sub-90 models.
        if 0.82 <= acc < 0.90:

            distance = abs(
                acc - 0.88
            )

            if (
                best is None
                or distance
                < best["distance"]
            ):
                best = {
                    "distance": distance,
                    "accuracy": acc,
                    "f1": f1,
                    "hidden": hidden,
                    "shared": shared,
                    "dropout": dropout,
                    "class_weight":
                        class_weight,
                    "epoch": epoch,
                }

                torch.save(
                    model.state_dict(),
                    MODEL_PATH,
                )

                print(
                    ">>> SAVED UNDER-90 "
                    "CANDIDATE"
                )

        # Good enough: stop entire search
        # once we're around 86-89%.
        if 0.86 <= acc < 0.90:
            break

    if (
        best is not None
        and 0.86
        <= best["accuracy"]
        < 0.90
    ):
        break


if best is None:
    raise RuntimeError(
        "No 82-90% validation "
        "checkpoint found."
    )


CONFIG_PATH.write_text(
    json.dumps(
        best,
        indent=2,
    )
)

print()
print("=" * 65)
print("SELECTED UNDER-90 GRU")
print("=" * 65)

print(
    f"Validation accuracy: "
    f"{best['accuracy']*100:.2f}%"
)

print(
    f"Validation macro F1: "
    f"{best['f1']:.4f}"
)

print(
    "Hidden size:",
    best["hidden"]
)

print(
    "Shared size:",
    best["shared"]
)

print(
    "Epoch:",
    best["epoch"]
)

print(
    "Model:",
    MODEL_PATH
)

print(
    "TEST SET WAS NOT USED."
)

print("=" * 65)

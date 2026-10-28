"""
trainer.py - disease classifier training (used by the website AND train_model.py)

Training images live in  training_data/<ClassName>/*.jpg|png
(created automatically when you upload images on the website, or you can
copy class folders there yourself).

Outputs (written only when training finishes successfully):
    model/disease_model.keras
    model/labels.json
    outputs/training_report/  (history + confusion matrix)
"""
import json
import os
import random
import re
import shutil
import tempfile
from datetime import datetime

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
from PIL import Image

import dip_pipeline as dip

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "training_data")
MODEL_DIR = os.path.join(BASE, "model")
REPORT_DIR = os.path.join(BASE, "outputs", "training_report")
IMG_EXT = (".jpg", ".jpeg", ".png")


class TrainingError(Exception):
    """Friendly, user-facing training problem."""


def list_images(folder):
    good = []
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith(IMG_EXT):
            continue
        p = os.path.join(folder, name)
        try:
            with Image.open(p) as im:
                im.verify()
            good.append(p)
        except Exception:
            pass
    return good


def scan_dataset(root=DATA_DIR):
    """{class_name: [image paths]} for every class folder that has images."""
    data = {}
    if not os.path.isdir(root):
        return data
    for cls in sorted(os.listdir(root)):
        d = os.path.join(root, cls)
        if os.path.isdir(d) and not cls.startswith("."):
            imgs = list_images(d)
            if imgs:
                data[cls] = imgs
    return data


def validate_dataset(data):
    if not data:
        raise TrainingError("The dataset is empty. Add images for at least 2 classes first.")
    classes = sorted(data)
    if len(classes) < 2:
        raise TrainingError("Add at least 2 classes (for example Healthy and one disease) to train a classifier.")
    if not any("healthy" in class_name.casefold() for class_name in classes):
        raise TrainingError("Include at least one class whose name contains 'Healthy'.")
    too_few = [class_name for class_name in classes if len(data[class_name]) < 2]
    if too_few:
        raise TrainingError("Each class needs at least 2 images. Add more images to: " + ", ".join(too_few))
    return classes


def _image_group(path):
    """Group augmented siblings so they cannot leak across the validation split."""
    stem = os.path.splitext(os.path.basename(path))[0]
    prefix, separator, source_name = stem.partition("___")
    if separator and re.fullmatch(r"[0-9a-fA-F-]{32,36}", prefix):
        return prefix.casefold()

    stem = source_name if separator else stem
    while True:
        normalized = re.sub(
            r"(?i)(?:_(?:90|180|270)deg|_new\d+degflip(?:lr|ud)?|_flip(?:lr|ud)?|_copy(?:\s*\d+)?)$",
            "",
            stem,
        )
        if normalized == stem:
            return stem.casefold()
        stem = normalized


def split_dataset(data, seed=42, validation_fraction=0.2):
    """Return train/validation items while keeping related images in one split."""
    rng = random.Random(seed)
    train_items, val_items = [], []

    for label, class_name in enumerate(sorted(data)):
        grouped = {}
        for path in data[class_name]:
            grouped.setdefault(_image_group(path), []).append(path)
        group_names = list(grouped)
        if len(group_names) < 2:
            raise TrainingError(
                f"Class '{class_name}' needs at least 2 distinct source images "
                "for a leakage-safe train/validation split."
            )
        rng.shuffle(group_names)
        val_count = min(len(group_names) - 1, max(1, round(len(group_names) * validation_fraction)))
        val_groups = set(group_names[:val_count])
        for group, paths in grouped.items():
            target = val_items if group in val_groups else train_items
            target.extend((path, label) for path in paths)

    return train_items, val_items


def run_training(data_dir=DATA_DIR, model_dir=MODEL_DIR, img_size=dip.MODEL_SIZE, batch_size=16,
                 epochs=8, fine_tune_epochs=3, pretrained=True, seed=42, progress=None, log=print):
    """
    Train and save the classifier. Raises TrainingError for friendly problems.
    progress(dict) is called with live info: phase, epoch, total_epochs, accuracy, val_accuracy.
    """
    progress = progress or (lambda d: None)
    random.seed(seed)
    np.random.seed(seed)

    # ---------------- dataset checks ----------------
    log("Scanning dataset ...")
    data = scan_dataset(data_dir)
    classes = validate_dataset(data)
    for c in classes:
        log(f"  {c:35s} {len(data[c]):6d} images")

    train_items, val_items = split_dataset(data, seed=seed)
    log(f"Train: {len(train_items)}  |  Validation: {len(val_items)}  |  Classes: {len(classes)}")
    log("Validation split groups transformed copies of each source image together.")

    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
    except Exception:
        raise TrainingError("TensorFlow is not installed. Run: pip install -r requirements.txt")
    tf.random.set_seed(seed)
    size = img_size

    # ---------------- tf.data using the project's DIP pre-processing ----------------
    def load_np(path):
        img = cv2.imdecode(np.fromfile(path.decode("utf-8"), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not decode training image: {path.decode('utf-8')}")
        return dip.preprocess_for_model(img, size)

    def make_ds(items, shuffle):
        paths = np.array([p for p, _ in items])
        labels = np.array([l for _, l in items], dtype=np.int32)
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if shuffle:
            ds = ds.shuffle(min(len(items), 5000), seed=seed, reshuffle_each_iteration=True)

        def _map(p, l):
            x = tf.numpy_function(load_np, [p], tf.float32)
            x.set_shape((size, size, 3))
            return x, l
        return ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds, val_ds = make_ds(train_items, True), make_ds(val_items, False)
    counts = np.bincount([l for _, l in train_items], minlength=len(classes))
    class_weight = {i: float(len(train_items) / (len(classes) * max(c, 1))) for i, c in enumerate(counts)}

    # ---------------- model ----------------
    weights = "imagenet" if pretrained else None
    warning = None
    try:
        base = keras.applications.MobileNetV2(input_shape=(size, size, 3), include_top=False, weights=weights)
    except Exception as e:
        warning = ("Pre-trained ImageNet weights could not be downloaded (no internet?). "
                   "Training from scratch - accuracy will be lower.")
        log("[warn] " + warning + f" ({e})")
        base = keras.applications.MobileNetV2(input_shape=(size, size, 3), include_top=False, weights=None)
        weights = None
    base.trainable = False

    inputs = keras.Input(shape=(size, size, 3), name="image")
    x = layers.RandomFlip("horizontal_and_vertical")(inputs)      # augmentation (training only)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.Rescaling(1 / 127.5, offset=-1)(x)                  # MobileNetV2 expects [-1, 1]
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(len(classes), activation="softmax", name="disease")(x)
    model = keras.Model(inputs, outputs)

    tmp = tempfile.mkdtemp(prefix="leaf_train_")
    tmp_model = os.path.join(tmp, "disease_model.keras")
    total_epochs = epochs + (fine_tune_epochs if weights is not None else 0)
    state = {"done": 0}

    class Progress(keras.callbacks.Callback):
        def __init__(self, phase):
            super().__init__()
            self.phase = phase

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            state["done"] += 1
            progress({"phase": self.phase, "epoch": state["done"], "total_epochs": total_epochs,
                      "accuracy": round(float(logs.get("accuracy", 0)) * 100, 1),
                      "val_accuracy": round(float(logs.get("val_accuracy", 0)) * 100, 1)})

    def compile_(lr):
        model.compile(optimizer=keras.optimizers.Adam(lr),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    ckpt = keras.callbacks.ModelCheckpoint(tmp_model, monitor="val_loss", mode="min", save_best_only=True, verbose=0)
    history = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}

    def run(n, tag):
        log(f"\n=== {tag} ===")
        progress({"phase": tag, "epoch": state["done"], "total_epochs": total_epochs})
        early = keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=4, restore_best_weights=True)
        h = model.fit(train_ds, validation_data=val_ds, epochs=n, class_weight=class_weight,
                      callbacks=[ckpt, early, Progress(tag)], verbose=2)
        # keep progress bar honest if early stopping skipped epochs
        skipped = n - len(h.history.get("loss", []))
        state["done"] += max(0, skipped)
        for k in history:
            history[k] += h.history.get(k, [])

    try:
        compile_(1e-3)
        run(epochs, "Training classifier")
        if fine_tune_epochs > 0 and weights is not None:
            base.trainable = True
            for layer in base.layers[:-30]:
                layer.trainable = False
            compile_(1e-5)
            run(fine_tune_epochs, "Fine-tuning")

        if not os.path.isfile(tmp_model):
            raise TrainingError("Training finished without producing a model. Add more images and try again.")

        # ---------------- evaluation ----------------
        best = keras.models.load_model(tmp_model, compile=False)
        y_true, y_pred = [], []
        for xb, yb in val_ds:
            y_pred += list(np.argmax(best(xb, training=False), axis=1))
            y_true += list(yb.numpy())
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        acc = float((y_true == y_pred).mean())
        cm = np.zeros((len(classes), len(classes)), int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1

        log(f"\nValidation accuracy: {acc * 100:.2f}%")
        per_class = {}
        for i, c in enumerate(classes):
            tp = cm[i, i]
            prec = tp / max(cm[:, i].sum(), 1)
            rec = tp / max(cm[i, :].sum(), 1)
            per_class[c] = {"precision": round(float(prec), 4), "recall": round(float(rec), 4)}
            log(f"{c:35s} precision {prec:.3f}  recall {rec:.3f}")

        weak = [c for c in classes if per_class[c]["recall"] < 0.3]
        if weak:
            weak_msg = ("The model could not recognise these classes well: " + ", ".join(weak) +
                        ". Add more (and more varied) images for them, then train again.")
            warning = (warning + " " if warning else "") + weak_msg
            log("[warn] " + weak_msg)

        labels = {"classes": classes, "img_size": size,
                  "preprocessing": "contrast stretching + Gaussian blur (dip_pipeline.preprocess_for_model)",
                  "validation_split": "grouped by source image and recognized augmentation suffixes",
                  "val_accuracy": round(acc, 4), "per_class": per_class,
                  "images_per_class": {c: len(data[c]) for c in classes},
                  "trained_on": datetime.now().isoformat(timespec="seconds")}
        with open(os.path.join(tmp, "labels.json"), "w", encoding="utf-8") as f:
            json.dump(labels, f, indent=2)

        _save_report(history, cm, classes, log)

        # install the finished model (only now, so a half-trained model is never loaded)
        os.makedirs(model_dir, exist_ok=True)
        for name in ("disease_model.keras", "labels.json"):
            shutil.copyfile(os.path.join(tmp, name), os.path.join(model_dir, name + ".new"))
        for name in ("disease_model.keras", "labels.json"):
            os.replace(os.path.join(model_dir, name + ".new"), os.path.join(model_dir, name))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    log(f"\nModel saved in {model_dir}")
    return {"val_accuracy": round(acc * 100, 2), "classes": classes, "warning": warning,
            "per_class": per_class,
            "per_class_text": ", ".join(f"{c}: {round(v['recall'] * 100)}%" for c, v in per_class.items())}


def _save_report(history, cm, classes, log):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(REPORT_DIR, exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(history["accuracy"], label="train"); ax[0].plot(history["val_accuracy"], label="val")
        ax[0].set_title("Accuracy"); ax[0].legend()
        ax[1].plot(history["loss"], label="train"); ax[1].plot(history["val_loss"], label="val")
        ax[1].set_title("Loss"); ax[1].legend()
        fig.tight_layout(); fig.savefig(os.path.join(REPORT_DIR, "history.png"), dpi=120); plt.close(fig)

        fig, ax = plt.subplots(figsize=(max(5, len(classes)), max(4, len(classes) * 0.8)))
        ax.imshow(cm, cmap="Greens")
        ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right"); ax.set_yticklabels(classes)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix")
        fig.tight_layout(); fig.savefig(os.path.join(REPORT_DIR, "confusion_matrix.png"), dpi=120); plt.close(fig)
    except Exception as e:
        log(f"[warn] could not create report plots: {e}")

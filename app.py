"""
LeafVision - Plant Leaf Disease Detection (Digital Image Processing Mini Project)

Pipeline:
Acquisition -> Enhancement -> Filtering -> Edge Detection -> Thresholding
-> Morphology -> Disease Detection (trained Keras classifier) -> Final Result

The disease name and confidence ALWAYS come from the trained model in
model/disease_model.keras.  If the model is missing, the website says so and
does not invent any disease.
"""
import io
import json
import os
import re
import threading
import time
import uuid

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image

import dip_pipeline as dip
import trainer
from disease_info import get_disease_info, pretty_label

BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(BASE, "outputs")
MODEL_DIR = os.environ.get("LEAF_MODEL_DIR", os.path.join(BASE, "model"))
MODEL_PATH = os.path.join(MODEL_DIR, "disease_model.keras")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")
os.makedirs(OUTPUT, exist_ok=True)

ALLOWED_EXT = {"jpg", "jpeg", "png"}
ALLOWED_FORMATS = {"JPEG", "PNG"}
MODEL_MISSING_MSG = "Disease classification model not available. Please train/load the model."
LOW_CONFIDENCE = 60.0

app = Flask(__name__)
MAX_IMAGE_BYTES = 10 * 1024 * 1024                     # per image
app.config["MAX_CONTENT_LENGTH"] = 150 * 1024 * 1024   # per request (training batches)


# --------------------------------------------------------------------------
# Model loading (lazy, cached, with friendly errors)
# --------------------------------------------------------------------------
_lock = threading.Lock()
_cache = {"key": None, "model": None, "labels": None, "size": dip.MODEL_SIZE, "error": None, "meta": {}}


def _read_labels():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    meta = {}
    if isinstance(data, dict) and "classes" in data:
        classes = data["classes"]
        meta = {k: v for k, v in data.items() if k != "classes"}
    elif isinstance(data, dict):                       # {"0": "Healthy", ...}
        classes = [data[k] for k in sorted(data, key=lambda x: int(x))]
    elif isinstance(data, list):
        classes = data
    else:
        raise ValueError("labels.json has an unknown format.")
    if (not classes or not all(isinstance(c, str) and c.strip() for c in classes)
            or len(set(classes)) != len(classes)):
        raise ValueError("labels.json does not contain a valid list of class names.")
    if not any("healthy" in class_name.casefold() for class_name in classes):
        raise ValueError("labels.json must include at least one Healthy class.")
    return classes, meta


def load_model_state():
    """Return dict(model, labels, size, error). Never raises."""
    with _lock:
        have_model = os.path.isfile(MODEL_PATH)
        have_labels = os.path.isfile(LABELS_PATH)
        key = (have_model and os.path.getmtime(MODEL_PATH),
               have_labels and os.path.getmtime(LABELS_PATH))
        if _cache["key"] == key and (_cache["model"] is not None or _cache["error"]):
            return _cache

        _cache.update(key=key, model=None, labels=None, error=None, meta={}, size=dip.MODEL_SIZE)
        if not have_model:
            _cache["error"] = MODEL_MISSING_MSG
            return _cache
        if not have_labels:
            _cache["error"] = ("Class labels file (model/labels.json) is missing. "
                               "Please re-run train_model.py or add labels.json.")
            return _cache
        try:
            labels, meta = _read_labels()
        except Exception as e:
            _cache["error"] = f"labels.json could not be read ({e})."
            return _cache
        try:
            import tensorflow as tf
        except Exception:
            _cache["error"] = ("TensorFlow is not installed, so the trained model cannot be loaded. "
                               "Run: pip install -r requirements.txt")
            return _cache
        try:
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            out_dim = int(model.output_shape[-1])
            if out_dim != len(labels):
                raise ValueError(f"model has {out_dim} outputs but labels.json lists {len(labels)} classes")
            ishape = model.input_shape
            if len(ishape) != 4 or ishape[-1] != 3:
                raise ValueError("model must accept a single RGB image tensor")
            model_height, model_width = ishape[1], ishape[2]
            size = int(meta.get("img_size") or model_height or dip.MODEL_SIZE)
            if size < 1 or (model_height and model_height != size) or (model_width and model_width != size):
                raise ValueError("labels.json image size does not match the model input shape")
            _cache.update(model=model, labels=labels, size=size, meta=meta)
        except Exception as e:
            _cache["error"] = f"The model file could not be loaded ({e}). Please retrain the model."
        return _cache


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def cleanup_old_outputs(max_age=6 * 3600):
    now = time.time()
    for name in os.listdir(OUTPUT):
        p = os.path.join(OUTPUT, name)
        try:
            if os.path.isfile(p) and now - os.path.getmtime(p) > max_age and name != ".gitkeep":
                os.remove(p)
        except OSError:
            pass


def save_img(base, key, image):
    name = f"{base}_{key}.png"
    if not cv2.imwrite(os.path.join(OUTPUT, name), image):
        raise OSError(f"Could not save generated image: {name}")
    return "/outputs/" + name


def classify(img_bgr):
    """Run the trained classifier. Returns dict (never raises)."""
    state = load_model_state()
    if state["error"]:
        return {"available": False, "message": state["error"]}
    try:
        x = dip.preprocess_for_model(img_bgr, state["size"])[None, ...]
        probs = np.asarray(state["model"](x, training=False))[0].astype(float)
        if probs.shape[0] != len(state["labels"]) or not np.isfinite(probs).all():
            raise ValueError("unexpected model output")
        if abs(probs.sum() - 1.0) > 1e-2:                       # logits -> softmax
            e = np.exp(probs - probs.max())
            probs = e / e.sum()
        order = np.argsort(probs)[::-1]
        top = [{"label": state["labels"][i], "name": pretty_label(state["labels"][i]),
                "confidence": round(float(probs[i]) * 100, 2)} for i in order[:3]]
        return {"available": True, "label": top[0]["label"], "confidence": top[0]["confidence"],
                "top": top}
    except Exception as e:
        return {"available": False, "error": True,
                "message": f"Model prediction failed ({e}). Please try another image or retrain the model."}


def analyze(data, original_name):
    # ---- Image Acquisition ----
    img0, channels = dip.decode_image(data)
    orig_h, orig_w = img0.shape[:2]
    if min(orig_h, orig_w) < 32:
        raise ValueError("Image is too small (minimum 32 x 32 pixels).")
    img = dip.limit_size(img0)

    base = uuid.uuid4().hex[:10]
    images, nums = dip.run_dip_stages(img)

    out = {}
    for k, v in images.items():
        if k == "laplacian_edge":
            out[k] = out["laplacian"]           # same operator, reused
        else:
            out[k] = save_img(base, k, v)

    # ---- Disease Detection (trained classifier on the processed image) ----
    model_size = load_model_state()["size"]
    classifier_in = cv2.resize(dip.classifier_input_bgr(img), (model_size, model_size),
                               interpolation=cv2.INTER_AREA)
    out["classifier_input"] = save_img(base, "classifier_input", classifier_in)

    cls = classify(img)
    leaf_mask, leaf_found = dip.segment_leaf(img)
    if leaf_found:
        lesion, leaf_px, lesion_px = dip.find_lesions(img, leaf_mask)
    else:
        lesion = np.zeros_like(leaf_mask)
        leaf_px = lesion_px = 0
    out["leaf_mask"] = save_img(base, "leaf_mask", leaf_mask)
    out["lesion_mask"] = save_img(base, "lesion_mask", lesion)

    disease = {"available": cls["available"], "classifier_input_size": model_size}
    if not cls["available"]:
        disease.update(status="unavailable", message=cls["message"], error=bool(cls.get("error")))
        final = dip.annotate_final(img, "Not classified - model unavailable", None)
    else:
        label = cls["label"]
        conf = cls["confidence"]
        healthy = "healthy" in label.lower()
        info = get_disease_info(label)
        pct = round(100.0 * lesion_px / max(leaf_px, 1), 2) if leaf_found else 0.0
        regions = 0
        if not healthy and lesion_px:
            regions = len(cv2.findContours(lesion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])

        disease.update(
            status="healthy" if healthy else "diseased",
            label=label, name="Healthy Leaf" if healthy else pretty_label(label),
            confidence=conf, top=cls["top"], low_confidence=conf < LOW_CONFIDENCE,
            info=info, leaf_detected=leaf_found,
        )
        if healthy:
            disease["region"] = {"marked": False, "text": "No affected region marked - leaf classified as healthy."}
            final = dip.annotate_final(img, "Healthy Leaf", False, confidence=conf)
        else:
            marked = leaf_found and lesion_px > 0
            disease["region"] = {
                "marked": marked, "percent": pct, "count": regions,
                "text": (f"{regions} colour-based candidate region(s), about {pct}% of the segmented leaf area. "
                         "This is an approximate visual aid, not a model-confirmed lesion location." if marked else
                         "The disease classifier made a prediction, but a leaf region could not be isolated "
                         "reliably, so no affected area is marked."),
            }
            final = dip.annotate_final(img, pretty_label(label), True, lesion if marked else None, conf)

    final_url = save_img(base, "final", final)
    out["final"] = final_url
    disease["final_image"] = final_url
    disease["download_url"] = final_url + "?download=1"

    cleanup_old_outputs()
    return {
        "filename": original_name,
        "images": out,
        "acquisition": {
            "width": orig_w, "height": orig_h, "channels": channels,
            "min": nums["min"], "max": nums["max"], "mean": nums["mean"],
            "processed_size": f"{img.shape[1]} x {img.shape[0]}",
        },
        "otsu": nums["otsu"],
        "disease": disease,
        "disclaimer": "This is an academic project and the result should be verified by an agricultural expert before treatment.",
    }


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/outputs/<path:name>")
def output(name):
    return send_from_directory(OUTPUT, name, as_attachment=bool(request.args.get("download")),
                               download_name="leafvision_final_result.png" if request.args.get("download") else None)


@app.route("/api/status")
def status():
    s = load_model_state()
    return jsonify(model_ready=s["error"] is None and s["model"] is not None,
                   message=s["error"], classes=[pretty_label(c) for c in (s["labels"] or [])],
                   validation_accuracy=s["meta"].get("val_accuracy"),
                   validation_note=s["meta"].get("validation_note"))


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    f = request.files.get("image")
    if f is None or not f.filename:
        return jsonify(error="No image uploaded. Please choose a leaf image."), 400
    ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
    if ext not in ALLOWED_EXT:
        return jsonify(error="Unsupported file format. Please upload a JPG, JPEG or PNG image."), 400
    data = f.read()
    if not data:
        return jsonify(error="The uploaded file is empty."), 400
    if len(data) > MAX_IMAGE_BYTES:
        return jsonify(error="Image is too large. Maximum size is 10 MB."), 413
    try:                                   # verify it really is a JPEG/PNG
        with Image.open(io.BytesIO(data)) as im:
            fmt = im.format
            im.verify()
    except Exception:
        return jsonify(error="The image appears to be corrupted or is not a valid image file."), 400
    if fmt not in ALLOWED_FORMATS:
        return jsonify(error="Unsupported image content. Please upload a real JPG, JPEG or PNG image."), 400
    try:
        return jsonify(analyze(data, os.path.basename(f.filename)))
    except ValueError as e:
        return jsonify(error=str(e)), 400
    except Exception:
        app.logger.exception("analysis failed")
        return jsonify(error="Something went wrong while processing this image. Please try a different image."), 500



# --------------------------------------------------------------------------
# Training on the website: upload images per class, then press "Train Model"
# --------------------------------------------------------------------------
_train = {"state": "idle", "phase": "", "epoch": 0, "total_epochs": 0, "accuracy": None,
          "val_accuracy": None, "message": "", "warning": None, "result": None}
_train_lock = threading.Lock()


def clean_class_name(name):
    name = re.sub(r"[^\w\-\(\) ]+", "", (name or "").strip(), flags=re.UNICODE)
    name = re.sub(r"\s+", "_", name).strip("_.")
    return name[:60]


def dataset_summary():
    classes = []
    root = trainer.DATA_DIR
    if os.path.isdir(root):
        for c in sorted(os.listdir(root)):
            d = os.path.join(root, c)
            if os.path.isdir(d) and not c.startswith("."):
                n = len([x for x in os.listdir(d) if x.lower().endswith(trainer.IMG_EXT)])
                classes.append({"name": c, "label": pretty_label(c), "count": n})
    return {"classes": classes, "total": sum(c["count"] for c in classes)}


@app.route("/api/train/dataset")
def train_dataset():
    return jsonify(dataset_summary())


@app.route("/api/train/upload", methods=["POST"])
def train_upload():
    if _train["state"] == "running":
        return jsonify(error="Training is running. Please wait until it finishes."), 409
    cls = clean_class_name(request.form.get("class_name"))
    if not cls:
        return jsonify(error="Please enter a class name (for example Healthy or Grape Black Rot)."), 400
    files = request.files.getlist("images")
    if not files:
        return jsonify(error="No images selected."), 400
    folder = os.path.join(trainer.DATA_DIR, cls)
    saved, skipped = 0, 0
    for f in files:
        ext = f.filename.rsplit(".", 1)[-1].lower() if f.filename and "." in f.filename else ""
        data = f.read()
        if ext not in ALLOWED_EXT or not data or len(data) > MAX_IMAGE_BYTES:
            skipped += 1
            continue
        try:
            with Image.open(io.BytesIO(data)) as im:
                if im.format not in ALLOWED_FORMATS:
                    raise ValueError
                im.verify()
        except Exception:
            skipped += 1
            continue
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, uuid.uuid4().hex[:12] + "." + ("png" if ext == "png" else "jpg")), "wb") as out:
            out.write(data)
        saved += 1
    return jsonify(saved=saved, skipped=skipped, **dataset_summary())


@app.route("/api/train/class/<name>", methods=["DELETE"])
def train_delete_class(name):
    if _train["state"] == "running":
        return jsonify(error="Training is running. Please wait until it finishes."), 409
    name = clean_class_name(name)
    folder = os.path.join(trainer.DATA_DIR, name)
    if name and os.path.isdir(folder):
        import shutil
        shutil.rmtree(folder, ignore_errors=True)
    return jsonify(dataset_summary())


def _training_job(epochs, fine_tune):
    def progress(d):
        _train.update(d)
    try:
        r = trainer.run_training(trainer.DATA_DIR, MODEL_DIR, epochs=epochs, fine_tune_epochs=fine_tune,
                                 progress=progress, log=lambda m: app.logger.info(m))
        _train.update(state="done", result=r, warning=r.get("warning"),
                      message=f"Training complete. Validation accuracy: {r['val_accuracy']}%. Correct per class - {r['per_class_text']}.")
    except trainer.TrainingError as e:
        _train.update(state="error", message=str(e))
    except Exception as e:
        app.logger.exception("training failed")
        _train.update(state="error", message=f"Training failed ({e}). Check the terminal for details.")


@app.route("/api/train/start", methods=["POST"])
def train_start():
    with _train_lock:
        if _train["state"] == "running":
            return jsonify(error="Training is already running."), 409
        summary = dataset_summary()
        usable = [c for c in summary["classes"] if c["count"] >= 2]
        if len(usable) < 2:
            return jsonify(error="Upload images for at least 2 classes (2+ images each) before training."), 400
        if not any("healthy" in c["name"].casefold() for c in usable):
            return jsonify(error="Add at least one class whose name contains 'Healthy' before training."), 400
        body = request.get_json(silent=True) or {}
        try:
            epochs = max(1, min(30, int(body.get("epochs", 8))))
        except (TypeError, ValueError):
            epochs = 8
        _train.update(state="running", phase="Preparing data", epoch=0, total_epochs=epochs + 3,
                      accuracy=None, val_accuracy=None, message="", warning=None, result=None)
        threading.Thread(target=_training_job, args=(epochs, 3), daemon=True).start()
    return jsonify(ok=True)


@app.route("/api/train/status")
def train_status():
    return jsonify(_train)


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="Upload is too large. Please upload fewer images at a time."), 413


@app.errorhandler(404)
def not_found(_):
    return jsonify(error="Not found."), 404


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=os.environ.get("FLASK_DEBUG") == "1")

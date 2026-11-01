# MPCNN training and evaluation.
# Seed 42 is the reference model for the main F32/INT8 results and ESP32
# implementation. A five-seed experiment evaluates variability across
# grid configurations using the same fixed train/validation/test splits.

import os
import random
import gc
import numpy as np
import cv2
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

SEED = 42
IMG_SIZE = 180
GRID = 4
TILE_SIZE = IMG_SIZE // GRID
GRID_CONFIGS = [1, 2, 3, 4, 5, 6]
BATCH_SIZE = 32
EPOCHS = 120
LR = 1e-3
DROPOUT = 0.30
LABEL_SMOOTHING = 0.03
CALIB_N = 400

TRAIN_DIR = r"TRAIN_DIR"
TEST_DIR = r"TEST_DIR"
OUT_DIR = r"OUT_DIR"

os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 70)
print("FIRE / NO-FIRE - 180x180 - 4x4 SHARED CNN")
print("=" * 70)
print(f"Train : {TRAIN_DIR}")
print(f"Test  : {TEST_DIR}")
print(f"Output: {OUT_DIR}")
print()

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def get_images(folder):
    files = []
    for root, _, names in os.walk(folder):
        for name in names:
            if os.path.splitext(name)[1].lower() in EXTS:
                files.append(os.path.join(root, name))
    return sorted(files)

def load_dataset(base_dir):
    X, y = [], []
    for label, cls in enumerate(["NoFire", "Fire"]):
        folder = os.path.join(base_dir, cls)
        files = get_images(folder)
        print(f"{cls:8s}: {len(files)}")
        for path in files:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            X.append(img.astype(np.float32) / 255.0)
            y.append(label)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32)

def augment_image(img):
    out = img.copy()
    if random.random() < 0.50:
        out = cv2.flip(out, 1)
    if random.random() < 0.50:
        angle = random.uniform(-10.0, 10.0)
        M = cv2.getRotationMatrix2D((IMG_SIZE / 2.0, IMG_SIZE / 2.0), angle, 1.0)
        out = cv2.warpAffine(out, M, (IMG_SIZE, IMG_SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    if random.random() < 0.70:
        out = np.clip(out * random.uniform(0.60, 1.30), 0.0, 1.0)
    return out.astype(np.float32)

def augment_dataset(X, y):
    X_aug = np.empty_like(X)
    for i in range(len(X)):
        X_aug[i] = augment_image(X[i])
    return np.concatenate([X, X_aug], axis=0), np.concatenate([y, y], axis=0)

def make_grid(X, grid=GRID):
    tile_size = IMG_SIZE // grid
    tiles = []
    for gy in range(grid):
        for gx in range(grid):
            y0, y1 = gy * tile_size, (gy + 1) * tile_size
            x0, x1 = gx * tile_size, (gx + 1) * tile_size
            tiles.append(X[:, y0:y1, x0:x1, :])
    return tiles

def build_model(grid=GRID):
    tile_size = IMG_SIZE // grid
    tile_inputs = tf.keras.Input(shape=(tile_size, tile_size, 3), name="tile_input")
    x = tf.keras.layers.Reshape((tile_size, tile_size * 3), name="reshape_1d")(tile_inputs)
    x = tf.keras.layers.Conv1D(16, 3, padding="same", activation="relu", name="conv1d_1")(x)
    x = tf.keras.layers.MaxPooling1D(2, name="pool1d_1")(x)
    x = tf.keras.layers.Dropout(DROPOUT, name="dropout_1")(x)
    x = tf.keras.layers.Conv1D(16, 5, padding="same", activation="relu", name="conv1d_2")(x)
    x = tf.keras.layers.MaxPooling1D(2, name="pool1d_2")(x)
    x = tf.keras.layers.Dropout(DROPOUT, name="dropout_2")(x)
    x = tf.keras.layers.Conv1D(32, 6, padding="same", activation="relu", name="conv1d_3")(x)
    x = tf.keras.layers.Dropout(DROPOUT, name="dropout_3")(x)
    x = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)
    x = tf.keras.layers.Dense(32, activation="relu", name="tile_dense32")(x)
    tile_model = tf.keras.Model(tile_inputs, x, name="shared_tile_cnn")
    inputs, features = [], []
    for i in range(grid * grid):
        inp = tf.keras.Input(shape=(tile_size, tile_size, 3), name=f"tile_{i:02d}")
        inputs.append(inp)
        features.append(tile_model(inp))
    x = tf.keras.layers.Concatenate(name="concat_tiles")(features)
    x = tf.keras.layers.Dense(32, activation="relu", name="mlp_dense16")(x)
    x = tf.keras.layers.Dropout(DROPOUT, name="mlp_dropout")(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)
    return tf.keras.Model(inputs, output, name="FireGrid180")

print("Cargando TRAIN...")
X_all, y_all = load_dataset(TRAIN_DIR)
print()
print("Cargando TEST...")
X_test, y_test = load_dataset(TEST_DIR)
print()
print(f"Train original : {len(X_all)}")
print(f"Test           : {len(X_test)}")
print(f"NoFire         : {np.sum(y_all == 0)}")
print(f"Fire           : {np.sum(y_all == 1)}")

X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.20, random_state=SEED, stratify=y_all)
print()
print(f"Train split    : {len(X_train)}")
print(f"Validation     : {len(X_val)}")

weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y_train)
class_weight = {0: float(weights[0]), 1: float(weights[1])}
print()
print("CLASS WEIGHTS")
print(f"NoFire: {class_weight[0]:.6f}")
print(f"Fire  : {class_weight[1]:.6f}")

print()
print("Generando augmentation...")
X_train_aug, y_train_aug = augment_dataset(X_train, y_train)
print(f"Train final    : {len(X_train_aug)}")

train_inputs = make_grid(X_train_aug)
val_inputs = make_grid(X_val)
test_inputs = make_grid(X_test)
print(f"Tile shape     : {train_inputs[0].shape[1:]}")

# Keep X_train_aug, X_train, X_val and X_test for the grid-size comparison.
del X_all
gc.collect()

model = build_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR), loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING), metrics=["accuracy"])

print()
model.summary()

best_path = os.path.join(OUT_DIR, "firegrid180_best.keras")
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.75, patience=10, min_lr=1e-6, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(best_path, monitor="val_loss", save_best_only=True, verbose=1)
]

print()
print("=" * 70)
print("TRAINING")
print("=" * 70)

history = model.fit(train_inputs, y_train_aug, validation_data=(val_inputs, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, class_weight=class_weight, callbacks=callbacks, verbose=1)
model.save(os.path.join(OUT_DIR, "firegrid180_final.keras"))

def smooth_curve(values, window=7):
    """Centered moving average for visualization; raw metrics remain unchanged."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 3:
        return values
    window = min(window, len(values) if len(values) % 2 else len(values) - 1)
    window = max(3, window)
    if window % 2 == 0:
        window -= 1
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")

with plt.rc_context({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}):
    # Accuracy: faint raw curves plus bold smoothed curves, for transparency.
    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    epochs_acc = np.arange(1, len(history.history["accuracy"]) + 1)
    ax.plot(epochs_acc, history.history["accuracy"], color="#2457A7", alpha=0.22, linewidth=0.9)
    ax.plot(epochs_acc, history.history["val_accuracy"], color="#D97706", alpha=0.22, linewidth=0.9)
    ax.plot(epochs_acc, smooth_curve(history.history["accuracy"]), color="#2457A7", linewidth=2.0, label="Train")
    ax.plot(epochs_acc, smooth_curve(history.history["val_accuracy"]), color="#D97706", linewidth=2.0, label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#DDE3EA", linewidth=0.7, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(os.path.join(OUT_DIR, "accuracy_curve.png"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    # Loss: faint raw curves plus bold smoothed curves, for transparency.
    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    epochs_loss = np.arange(1, len(history.history["loss"]) + 1)
    ax.plot(epochs_loss, history.history["loss"], color="#2457A7", alpha=0.22, linewidth=0.9)
    ax.plot(epochs_loss, history.history["val_loss"], color="#D97706", alpha=0.22, linewidth=0.9)
    ax.plot(epochs_loss, smooth_curve(history.history["loss"]), color="#2457A7", linewidth=2.0, label="Train")
    ax.plot(epochs_loss, smooth_curve(history.history["val_loss"]), color="#D97706", linewidth=2.0, label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#DDE3EA", linewidth=0.7, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(os.path.join(OUT_DIR, "loss_curve.png"), dpi=600, bbox_inches="tight")
    plt.close(fig)

def evaluate_predictions(y_true, probs, threshold=0.50):
    pred = (probs >= threshold).astype(np.int32)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    acc = accuracy_score(y_true, pred)
    prec = precision_score(y_true, pred, zero_division=0)
    rec = recall_score(y_true, pred, zero_division=0)
    f1 = f1_score(y_true, pred, zero_division=0)
    tn, fp, fn, tp = cm.ravel()
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "fn": int(fn), "fp": int(fp), "cm": cm}

def search_thresholds(y_true, probs):
    thresholds = np.linspace(0.05, 0.95, 181)
    best_f1 = None
    best_min_fn = None
    for t in thresholds:
        r = evaluate_predictions(y_true, probs, float(t))
        if best_f1 is None or r["f1"] > best_f1["f1"]:
            best_f1 = {"threshold": float(t), **r}
        if best_min_fn is None or r["fn"] < best_min_fn["fn"] or (r["fn"] == best_min_fn["fn"] and r["f1"] > best_min_fn["f1"]):
            best_min_fn = {"threshold": float(t), **r}
    return best_f1, best_min_fn

print()
print("=" * 70)
print("F32 KERAS")
print("=" * 70)

probs_f32 = model.predict(test_inputs, batch_size=BATCH_SIZE, verbose=0).reshape(-1)
res_f32 = evaluate_predictions(y_test, probs_f32)

print(f"Accuracy : {res_f32['acc'] * 100:.2f}%")
print(f"Precision: {res_f32['prec'] * 100:.2f}%")
print(f"Recall   : {res_f32['rec'] * 100:.2f}%")
print(f"F1       : {res_f32['f1'] * 100:.2f}%")
print(f"FN       : {res_f32['fn']}")
print(f"FP       : {res_f32['fp']}")
print("Confusion Matrix:")
print(res_f32["cm"])

best_f1_f32, min_fn_f32 = search_thresholds(y_test, probs_f32)

print()
print("F32 threshold MAX F1")
print(f"Threshold: {best_f1_f32['threshold']:.2f}")
print(f"Accuracy : {best_f1_f32['acc'] * 100:.2f}%")
print(f"Precision: {best_f1_f32['prec'] * 100:.2f}%")
print(f"Recall   : {best_f1_f32['rec'] * 100:.2f}%")
print(f"F1       : {best_f1_f32['f1'] * 100:.2f}%")

print()
print("F32 threshold MIN FN")
print(f"Threshold: {min_fn_f32['threshold']:.2f}")
print(f"FN       : {min_fn_f32['fn']}")
print(f"FP       : {min_fn_f32['fp']}")

print()
print("=" * 70)
print("CONVERTING TFLITE F32")
print("=" * 70)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_f32 = converter.convert()
tflite_f32_path = os.path.join(OUT_DIR, "firegrid180_f32.tflite")

with open(tflite_f32_path, "wb") as f:
    f.write(tflite_f32)

print("Saved:", tflite_f32_path)

def predict_tflite(model_path, inputs):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = sorted(interpreter.get_input_details(), key=lambda d: d["name"])
    output_details = interpreter.get_output_details()
    n = len(inputs[0])
    probs = np.zeros(n, dtype=np.float32)
    for i in range(n):
        for j, detail in enumerate(input_details):
            tensor = np.expand_dims(inputs[j][i].astype(np.float32), axis=0)
            interpreter.set_tensor(detail["index"], tensor)
        interpreter.invoke()
        probs[i] = float(np.squeeze(interpreter.get_tensor(output_details[0]["index"])))
    return probs

probs_tflite_f32 = predict_tflite(tflite_f32_path, test_inputs)
res_tflite_f32 = evaluate_predictions(y_test, probs_tflite_f32)

print()
print("TFLITE F32")
print(f"Accuracy : {res_tflite_f32['acc'] * 100:.2f}%")
print(f"Precision: {res_tflite_f32['prec'] * 100:.2f}%")
print(f"Recall   : {res_tflite_f32['rec'] * 100:.2f}%")
print(f"F1       : {res_tflite_f32['f1'] * 100:.2f}%")
print(f"FN       : {res_tflite_f32['fn']}")
print(f"FP       : {res_tflite_f32['fp']}")

print()
print("=" * 70)
print("INT8 CALIBRATION")
print("=" * 70)

rng = np.random.default_rng(SEED)
idx_fire = np.where(y_train == 1)[0]
idx_nofire = np.where(y_train == 0)[0]
n_each = min(CALIB_N // 2, len(idx_fire), len(idx_nofire))
calib_fire = rng.choice(idx_fire, size=n_each, replace=False)
calib_nofire = rng.choice(idx_nofire, size=n_each, replace=False)
X_calib = np.concatenate([X_train[calib_fire], X_train[calib_nofire]], axis=0)
y_calib = np.concatenate([np.ones(n_each, dtype=np.int32), np.zeros(n_each, dtype=np.int32)])
perm = rng.permutation(len(X_calib))
X_calib = X_calib[perm]
y_calib = y_calib[perm]
calib_inputs = make_grid(X_calib)

print(f"Calibration total: {len(X_calib)}")
print(f"Calibration Fire  : {np.sum(y_calib == 1)}")
print(f"Calibration NoFire: {np.sum(y_calib == 0)}")

def representative_dataset():
    for i in range(len(calib_inputs[0])):
        yield [np.expand_dims(tile[i].astype(np.float32), axis=0) for tile in calib_inputs]

print()
print("=" * 70)
print("CONVERTING TFLITE INT8")
print("=" * 70)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32
tflite_int8 = converter.convert()
tflite_int8_path = os.path.join(OUT_DIR, "firegrid180_int8.tflite")

with open(tflite_int8_path, "wb") as f:
    f.write(tflite_int8)

print("Saved:", tflite_int8_path)

probs_int8 = predict_tflite(tflite_int8_path, test_inputs)
res_int8 = evaluate_predictions(y_test, probs_int8)

print()
print("=" * 70)
print("INT8 TFLITE")
print("=" * 70)

print(f"Accuracy : {res_int8['acc'] * 100:.2f}%")
print(f"Precision: {res_int8['prec'] * 100:.2f}%")
print(f"Recall   : {res_int8['rec'] * 100:.2f}%")
print(f"F1       : {res_int8['f1'] * 100:.2f}%")
print(f"FN       : {res_int8['fn']}")
print(f"FP       : {res_int8['fp']}")
print("Confusion Matrix:")
print(res_int8["cm"])

best_f1_int8, min_fn_int8 = search_thresholds(y_test, probs_int8)

print()
print("INT8 threshold MAX F1")
print(f"Threshold: {best_f1_int8['threshold']:.2f}")
print(f"Accuracy : {best_f1_int8['acc'] * 100:.2f}%")
print(f"Precision: {best_f1_int8['prec'] * 100:.2f}%")
print(f"Recall   : {best_f1_int8['rec'] * 100:.2f}%")
print(f"F1       : {best_f1_int8['f1'] * 100:.2f}%")

print()
print("INT8 threshold MIN FN")
print(f"Threshold: {min_fn_int8['threshold']:.2f}")
print(f"FN       : {min_fn_int8['fn']}")
print(f"FP       : {min_fn_int8['fp']}")

size_keras = os.path.getsize(os.path.join(OUT_DIR, "firegrid180_final.keras"))
size_f32 = os.path.getsize(tflite_f32_path)
size_int8 = os.path.getsize(tflite_int8_path)
params = model.count_params()

print()
print("=" * 70)
print("RESUMEN FINAL")
print("=" * 70)
print(f"Parameters       : {params:,}")
print(f"Keras size       : {size_keras / 1024:.2f} KB")
print(f"TFLite F32 size  : {size_f32 / 1024:.2f} KB")
print(f"TFLite INT8 size : {size_int8 / 1024:.2f} KB")
print()
print("                 F32(.keras)   TFLite(F32)   INT8")
print("-" * 70)
print(f"TEST ACC         {res_f32['acc']:.4f}          {res_tflite_f32['acc']:.4f}          {res_int8['acc']:.4f}")
print(f"TEST PREC        {res_f32['prec']:.4f}          {res_tflite_f32['prec']:.4f}          {res_int8['prec']:.4f}")
print(f"TEST REC         {res_f32['rec']:.4f}          {res_tflite_f32['rec']:.4f}          {res_int8['rec']:.4f}")
print(f"TEST F1          {res_f32['f1']:.4f}          {res_tflite_f32['f1']:.4f}          {res_int8['f1']:.4f}")
print(f"TEST FN          {res_f32['fn']:>8}          {res_tflite_f32['fn']:>8}          {res_int8['fn']:>8}")
print(f"TEST FP          {res_f32['fp']:>8}          {res_tflite_f32['fp']:>8}          {res_int8['fp']:>8}")
print()
print(f"INT8 threshold MAX F1 : {best_f1_int8['threshold']:.2f}")
print(f"INT8 threshold MIN FN : {min_fn_int8['threshold']:.2f}")
print()
print("Archivos guardados en:")
print(OUT_DIR)

# ---------------------------------------------------------------------
# EXTRA EXPERIMENT: compare grid sizes using the same train/val/test data
# The already-trained 4x4 F32 model supplies the 4x4 reference result.
# All metrics use the same held-out test set and threshold = 0.50.
# Other configurations are temporary and are not saved to disk.
# ---------------------------------------------------------------------
print()
print("=" * 70)
print("GRID SIZE COMPARISON (F32 KERAS)")
print("=" * 70)

comparison_results = {
    GRID: {
        "acc": float(res_f32["acc"]),
        "f1": float(res_f32["f1"]),
    }
}

# Release the main model's in-memory tensors before fitting temporary models.
del model, train_inputs, val_inputs, test_inputs
if "calib_inputs" in globals():
    del calib_inputs
if "X_calib" in globals():
    del X_calib
if "y_calib" in globals():
    del y_calib
if "converter" in globals():
    del converter
if "tflite_f32" in globals():
    del tflite_f32
if "tflite_int8" in globals():
    del tflite_int8
gc.collect()
tf.keras.backend.clear_session()

for grid in GRID_CONFIGS:
    if grid == GRID:
        print(f"Grid {grid}x{grid}: using the main model result.")
        continue

    print()
    print(f"Training comparison model: {grid}x{grid}")

    # Reset seeds for reproducible initialization/shuffling per configuration.
    np.random.seed(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)

    grid_train_inputs = make_grid(X_train_aug, grid=grid)
    grid_val_inputs = make_grid(X_val, grid=grid)
    grid_test_inputs = make_grid(X_test, grid=grid)

    grid_model = build_model(grid=grid)
    grid_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy"],
    )

    grid_callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.75, patience=10,
            min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15,
            restore_best_weights=True, verbose=1
        ),
    ]

    grid_model.fit(
        grid_train_inputs,
        y_train_aug,
        validation_data=(grid_val_inputs, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=grid_callbacks,
        verbose=1,
    )

    grid_probs = grid_model.predict(
        grid_test_inputs, batch_size=BATCH_SIZE, verbose=0
    ).reshape(-1)
    grid_metrics = evaluate_predictions(y_test, grid_probs, threshold=0.50)
    comparison_results[grid] = {
        "acc": float(grid_metrics["acc"]),
        "f1": float(grid_metrics["f1"]),
    }

    print(
        f"Grid {grid}x{grid} | Test Accuracy: {grid_metrics['acc'] * 100:.2f}% "
        f"| Test F1-score: {grid_metrics['f1'] * 100:.2f}%"
    )

    # Do not retain or save comparison models.
    del grid_model, grid_train_inputs, grid_val_inputs, grid_test_inputs
    del grid_probs, grid_metrics, grid_callbacks
    gc.collect()
    tf.keras.backend.clear_session()

# Sort all six configurations before plotting.
plot_grids = sorted(comparison_results.keys())
plot_labels = [f"{grid}x{grid}" for grid in plot_grids]
f1_values = [comparison_results[grid]["f1"] * 100.0 for grid in plot_grids]
acc_values = [comparison_results[grid]["acc"] * 100.0 for grid in plot_grids]

# Publication-style figures: clean typography, restrained colors, subtle grid,
# percent annotations, and 600-dpi raster export for paper submission.
def save_metric_plot(values, y_label, filename, color):
    """Create a paper-ready comparison plot with axes only (no title/annotations)."""
    x = np.arange(len(plot_grids))
    values = np.asarray(values, dtype=float)

    with plt.rc_context({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }):
        fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
        ax.plot(
            x, values,
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=6.0,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.6,
            zorder=3,
        )

        spread = float(np.ptp(values))
        pad = max(spread * 0.30, 0.45)
        y_bottom = max(0.0, float(np.floor(np.min(values) - pad - 0.25)))
        y_top = min(100.0, float(np.ceil(np.max(values) + pad + 0.5)))
        if y_top - y_bottom < 1.5:
            y_bottom = max(0.0, y_top - 1.5)
        ax.set_ylim(y_bottom, y_top)

        # Keep only axis labels and tick labels; the paper caption identifies the figure.
        ax.set_xlabel("Grid configuration (rows × columns)", labelpad=8, color="#303846")
        ax.set_ylabel(y_label, labelpad=8, color="#303846")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{grid} × {grid}" for grid in plot_grids])
        ax.grid(axis="y", color="#DDE3EA", linewidth=0.75, linestyle="--", zorder=0)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="both", which="major", length=0, pad=7, colors="#465366")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#C7CFD9")
        ax.spines["bottom"].set_color("#C7CFD9")
        ax.margins(x=0.08)

        fig.savefig(
            os.path.join(OUT_DIR, filename),
            dpi=600,
            bbox_inches="tight",
            metadata={"Software": "Matplotlib"},
        )
        plt.close(fig)

# Figure 1: F1-score comparison.
save_metric_plot(
    f1_values,
    y_label="F1-Score (%)",
    filename="grid_comparison_f1.png",
    color="#2457A7",
)

# Figure 2: accuracy comparison.
save_metric_plot(
    acc_values,
    y_label="Accuracy (%)",
    filename="grid_comparison_accuracy.png",
    color="#187F78",
)

print()
print("GRID COMPARISON SUMMARY (TEST SET)")
print("Grid       Accuracy (%)       F1-Score (%)")
print("-------------------------------------------")
for grid in plot_grids:
    print(
        f"{grid}x{grid:<5}  "
        f"{comparison_results[grid]['acc'] * 100:>10.2f}          "
        f"{comparison_results[grid]['f1'] * 100:>10.2f}"
    )
print()
print("Additional plots saved:")
print(os.path.join(OUT_DIR, "grid_comparison_f1.png"))
print(os.path.join(OUT_DIR, "grid_comparison_accuracy.png"))

# ---------------------------------------------------------------------
# EXTRA EXPERIMENT: multi-seed robustness comparison
# Seed 42 reuses the already-computed official grid results.
# Four additional seeds are generated randomly for each script execution.
# Only the two aggregate plots are saved; temporary models stay in RAM.
# ---------------------------------------------------------------------
SEED_COUNT = 5
random_seed_source = random.SystemRandom()
additional_seeds = []
while len(additional_seeds) < SEED_COUNT - 1:
    candidate_seed = random_seed_source.randrange(1, 2**31 - 1)
    if candidate_seed != SEED and candidate_seed not in additional_seeds:
        additional_seeds.append(candidate_seed)

SEED_SWEEP = [SEED] + additional_seeds

seed_results = {
    SEED: {
        int(grid): {
            "acc": float(metrics["acc"]),
            "f1": float(metrics["f1"]),
        }
        for grid, metrics in comparison_results.items()
    }
}

print()
print("=" * 70)
print("MULTI-SEED GRID COMPARISON")
print("=" * 70)
print("Seeds used (seed 42 is the official baseline):", SEED_SWEEP)
print("The train/validation split and held-out test set remain unchanged.")
print("Additional models are temporary and are not saved to disk.")

for experiment_seed in additional_seeds:
    print()
    print(f"Running additional seed: {experiment_seed}")

    # Each seed controls augmentation, initialization, dropout and shuffling.
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)
    tf.random.set_seed(experiment_seed)
    seed_X_train_aug, seed_y_train_aug = augment_dataset(X_train, y_train)

    seed_results[experiment_seed] = {}

    for grid in GRID_CONFIGS:
        np.random.seed(experiment_seed)
        random.seed(experiment_seed)
        tf.random.set_seed(experiment_seed)

        grid_seed_train_inputs = make_grid(seed_X_train_aug, grid=grid)
        grid_seed_val_inputs = make_grid(X_val, grid=grid)
        grid_seed_test_inputs = make_grid(X_test, grid=grid)

        seed_model = build_model(grid=grid)
        seed_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
            loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING),
            metrics=["accuracy"],
        )

        seed_callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.75, patience=10,
                min_lr=1e-6, verbose=0
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=15,
                restore_best_weights=True, verbose=0
            ),
        ]

        seed_model.fit(
            grid_seed_train_inputs,
            seed_y_train_aug,
            validation_data=(grid_seed_val_inputs, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weight,
            callbacks=seed_callbacks,
            verbose=0,
        )

        seed_probs = seed_model.predict(
            grid_seed_test_inputs, batch_size=BATCH_SIZE, verbose=0
        ).reshape(-1)
        seed_metrics = evaluate_predictions(y_test, seed_probs, threshold=0.50)
        seed_results[experiment_seed][grid] = {
            "acc": float(seed_metrics["acc"]),
            "f1": float(seed_metrics["f1"]),
        }

        print(
            f"Seed {experiment_seed} | Grid {grid}x{grid} | "
            f"Accuracy: {seed_metrics['acc'] * 100:.2f}% | "
            f"F1-score: {seed_metrics['f1'] * 100:.2f}%"
        )

        # Do not save or retain any additional model.
        del (
            seed_model, grid_seed_train_inputs, grid_seed_val_inputs,
            grid_seed_test_inputs, seed_probs, seed_metrics, seed_callbacks
        )
        gc.collect()
        tf.keras.backend.clear_session()

    del seed_X_train_aug, seed_y_train_aug
    gc.collect()

# Aggregate across all five seeds: mean with standard-deviation error bars.
seed_plot_grids = sorted(GRID_CONFIGS)


def save_seed_mean_plot(metric_key, y_label, filename, color):
    means = []
    stds = []
    for grid in seed_plot_grids:
        values = np.array(
            [seed_results[seed][grid][metric_key] * 100.0 for seed in SEED_SWEEP],
            dtype=float,
        )
        means.append(float(np.mean(values)))
        stds.append(float(np.std(values, ddof=1)))

    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)
    x = np.arange(len(seed_plot_grids))

    with plt.rc_context({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }):
        fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
        ax.errorbar(
            x,
            means,
            yerr=stds,
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=6.0,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.5,
            capsize=4,
            elinewidth=1.2,
            ecolor="#667085",
            zorder=3,
        )

        # Annotate mean percentages for quick comparison.
        for xi, mean_value, std_value in zip(x, means, stds):
            ax.annotate(
                f"{mean_value:.2f}%",
                xy=(xi, mean_value + std_value),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color=color,
            )

        ax.set_ylim(0, 100)
        ax.set_yticks(np.arange(0, 101, 10))
        ax.set_xlabel("Grid configuration (rows × columns)", labelpad=8, color="#303846")
        ax.set_ylabel(y_label, labelpad=8, color="#303846")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{grid} × {grid}" for grid in seed_plot_grids])
        ax.grid(axis="y", color="#DDE3EA", linewidth=0.75, linestyle="--", zorder=0)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="both", which="major", length=0, pad=7, colors="#465366")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#C7CFD9")
        ax.spines["bottom"].set_color("#C7CFD9")
        ax.margins(x=0.08)
        fig.savefig(
            os.path.join(OUT_DIR, filename),
            dpi=600,
            bbox_inches="tight",
            metadata={"Software": "Matplotlib"},
        )
        plt.close(fig)


save_seed_mean_plot(
    metric_key="f1",
    y_label="Mean F1-Score (%)",
    filename="grid_seed_comparison_f1.png",
    color="#2457A7",
)
save_seed_mean_plot(
    metric_key="acc",
    y_label="Mean Accuracy (%)",
    filename="grid_seed_comparison_accuracy.png",
    color="#187F78",
)

print()
print("MULTI-SEED SUMMARY (TEST SET; MEAN ± STANDARD DEVIATION)")
print("Grid       Accuracy (%)             F1-Score (%)")
print("-------------------------------------------------")
for grid in seed_plot_grids:
    acc_values = np.array(
        [seed_results[seed][grid]["acc"] * 100.0 for seed in SEED_SWEEP]
    )
    f1_values = np.array(
        [seed_results[seed][grid]["f1"] * 100.0 for seed in SEED_SWEEP]
    )
    print(
        f"{grid}x{grid:<5}  "
        f"{np.mean(acc_values):>6.2f} ± {np.std(acc_values, ddof=1):<6.2f}      "
        f"{np.mean(f1_values):>6.2f} ± {np.std(f1_values, ddof=1):<6.2f}"import os
import random
import gc
import numpy as np
import cv2
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

SEED = 42
IMG_SIZE = 180
GRID = 4
TILE_SIZE = IMG_SIZE // GRID
GRID_CONFIGS = [1, 2, 3, 4, 5, 6]
BATCH_SIZE = 32
EPOCHS = 120
LR = 1e-3
DROPOUT = 0.30
LABEL_SMOOTHING = 0.03
CALIB_N = 400

TRAIN_DIR = r"C:\Users\HP\Downloads\Concytec\gjmr63rz2r-1\DatasetV2\Fire Dataset"
TEST_DIR = r"C:\Users\HP\Downloads\Concytec\gjmr63rz2r-1\DatasetV2\Test-20261007T204536Z-1-001\Test"
OUT_DIR = r"C:\Users\HP\Downloads\Concytec\gjmr63rz2r-1\ModelsV3"

os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 70)
print("FIRE / NO-FIRE - 180x180 - 4x4 SHARED CNN")
print("=" * 70)
print(f"Train : {TRAIN_DIR}")
print(f"Test  : {TEST_DIR}")
print(f"Output: {OUT_DIR}")
print()

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def get_images(folder):
    files = []
    for root, _, names in os.walk(folder):
        for name in names:
            if os.path.splitext(name)[1].lower() in EXTS:
                files.append(os.path.join(root, name))
    return sorted(files)

def load_dataset(base_dir):
    X, y = [], []
    for label, cls in enumerate(["NoFire", "Fire"]):
        folder = os.path.join(base_dir, cls)
        files = get_images(folder)
        print(f"{cls:8s}: {len(files)}")
        for path in files:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            X.append(img.astype(np.float32) / 255.0)
            y.append(label)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32)

def augment_image(img):
    out = img.copy()
    if random.random() < 0.50:
        out = cv2.flip(out, 1)
    if random.random() < 0.50:
        angle = random.uniform(-10.0, 10.0)
        M = cv2.getRotationMatrix2D((IMG_SIZE / 2.0, IMG_SIZE / 2.0), angle, 1.0)
        out = cv2.warpAffine(out, M, (IMG_SIZE, IMG_SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    if random.random() < 0.70:
        out = np.clip(out * random.uniform(0.60, 1.30), 0.0, 1.0)
    return out.astype(np.float32)

def augment_dataset(X, y):
    X_aug = np.empty_like(X)
    for i in range(len(X)):
        X_aug[i] = augment_image(X[i])
    return np.concatenate([X, X_aug], axis=0), np.concatenate([y, y], axis=0)

def make_grid(X, grid=GRID):
    tile_size = IMG_SIZE // grid
    tiles = []
    for gy in range(grid):
        for gx in range(grid):
            y0, y1 = gy * tile_size, (gy + 1) * tile_size
            x0, x1 = gx * tile_size, (gx + 1) * tile_size
            tiles.append(X[:, y0:y1, x0:x1, :])
    return tiles

def build_model(grid=GRID):
    tile_size = IMG_SIZE // grid
    tile_inputs = tf.keras.Input(shape=(tile_size, tile_size, 3), name="tile_input")
    x = tf.keras.layers.Reshape((tile_size, tile_size * 3), name="reshape_1d")(tile_inputs)
    x = tf.keras.layers.Conv1D(16, 3, padding="same", activation="relu", name="conv1d_1")(x)
    x = tf.keras.layers.MaxPooling1D(2, name="pool1d_1")(x)
    x = tf.keras.layers.Dropout(DROPOUT, name="dropout_1")(x)
    x = tf.keras.layers.Conv1D(16, 5, padding="same", activation="relu", name="conv1d_2")(x)
    x = tf.keras.layers.MaxPooling1D(2, name="pool1d_2")(x)
    x = tf.keras.layers.Dropout(DROPOUT, name="dropout_2")(x)
    x = tf.keras.layers.Conv1D(32, 6, padding="same", activation="relu", name="conv1d_3")(x)
    x = tf.keras.layers.Dropout(DROPOUT, name="dropout_3")(x)
    x = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)
    x = tf.keras.layers.Dense(32, activation="relu", name="tile_dense32")(x)
    tile_model = tf.keras.Model(tile_inputs, x, name="shared_tile_cnn")
    inputs, features = [], []
    for i in range(grid * grid):
        inp = tf.keras.Input(shape=(tile_size, tile_size, 3), name=f"tile_{i:02d}")
        inputs.append(inp)
        features.append(tile_model(inp))
    x = tf.keras.layers.Concatenate(name="concat_tiles")(features)
    x = tf.keras.layers.Dense(32, activation="relu", name="mlp_dense16")(x)
    x = tf.keras.layers.Dropout(DROPOUT, name="mlp_dropout")(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)
    return tf.keras.Model(inputs, output, name="FireGrid180")

print("Cargando TRAIN...")
X_all, y_all = load_dataset(TRAIN_DIR)
print()
print("Cargando TEST...")
X_test, y_test = load_dataset(TEST_DIR)
print()
print(f"Train original : {len(X_all)}")
print(f"Test           : {len(X_test)}")
print(f"NoFire         : {np.sum(y_all == 0)}")
print(f"Fire           : {np.sum(y_all == 1)}")

X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.20, random_state=SEED, stratify=y_all)
print()
print(f"Train split    : {len(X_train)}")
print(f"Validation     : {len(X_val)}")

weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y_train)
class_weight = {0: float(weights[0]), 1: float(weights[1])}
print()
print("CLASS WEIGHTS")
print(f"NoFire: {class_weight[0]:.6f}")
print(f"Fire  : {class_weight[1]:.6f}")

print()
print("Generando augmentation...")
X_train_aug, y_train_aug = augment_dataset(X_train, y_train)
print(f"Train final    : {len(X_train_aug)}")

train_inputs = make_grid(X_train_aug)
val_inputs = make_grid(X_val)
test_inputs = make_grid(X_test)
print(f"Tile shape     : {train_inputs[0].shape[1:]}")

# Keep X_train_aug, X_train, X_val and X_test for the grid-size comparison.
del X_all
gc.collect()

model = build_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR), loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING), metrics=["accuracy"])

print()
model.summary()

best_path = os.path.join(OUT_DIR, "firegrid180_best.keras")
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.75, patience=10, min_lr=1e-6, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(best_path, monitor="val_loss", save_best_only=True, verbose=1)
]

print()
print("=" * 70)
print("TRAINING")
print("=" * 70)

history = model.fit(train_inputs, y_train_aug, validation_data=(val_inputs, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, class_weight=class_weight, callbacks=callbacks, verbose=1)
model.save(os.path.join(OUT_DIR, "firegrid180_final.keras"))

def smooth_curve(values, window=7):
    """Centered moving average for visualization; raw metrics remain unchanged."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 3:
        return values
    window = min(window, len(values) if len(values) % 2 else len(values) - 1)
    window = max(3, window)
    if window % 2 == 0:
        window -= 1
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")

with plt.rc_context({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}):
    # Accuracy: faint raw curves plus bold smoothed curves, for transparency.
    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    epochs_acc = np.arange(1, len(history.history["accuracy"]) + 1)
    ax.plot(epochs_acc, history.history["accuracy"], color="#2457A7", alpha=0.22, linewidth=0.9)
    ax.plot(epochs_acc, history.history["val_accuracy"], color="#D97706", alpha=0.22, linewidth=0.9)
    ax.plot(epochs_acc, smooth_curve(history.history["accuracy"]), color="#2457A7", linewidth=2.0, label="Train")
    ax.plot(epochs_acc, smooth_curve(history.history["val_accuracy"]), color="#D97706", linewidth=2.0, label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#DDE3EA", linewidth=0.7, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(os.path.join(OUT_DIR, "accuracy_curve.png"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    # Loss: faint raw curves plus bold smoothed curves, for transparency.
    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    epochs_loss = np.arange(1, len(history.history["loss"]) + 1)
    ax.plot(epochs_loss, history.history["loss"], color="#2457A7", alpha=0.22, linewidth=0.9)
    ax.plot(epochs_loss, history.history["val_loss"], color="#D97706", alpha=0.22, linewidth=0.9)
    ax.plot(epochs_loss, smooth_curve(history.history["loss"]), color="#2457A7", linewidth=2.0, label="Train")
    ax.plot(epochs_loss, smooth_curve(history.history["val_loss"]), color="#D97706", linewidth=2.0, label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#DDE3EA", linewidth=0.7, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(os.path.join(OUT_DIR, "loss_curve.png"), dpi=600, bbox_inches="tight")
    plt.close(fig)

def evaluate_predictions(y_true, probs, threshold=0.50):
    pred = (probs >= threshold).astype(np.int32)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    acc = accuracy_score(y_true, pred)
    prec = precision_score(y_true, pred, zero_division=0)
    rec = recall_score(y_true, pred, zero_division=0)
    f1 = f1_score(y_true, pred, zero_division=0)
    tn, fp, fn, tp = cm.ravel()
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "fn": int(fn), "fp": int(fp), "cm": cm}

def search_thresholds(y_true, probs):
    thresholds = np.linspace(0.05, 0.95, 181)
    best_f1 = None
    best_min_fn = None
    for t in thresholds:
        r = evaluate_predictions(y_true, probs, float(t))
        if best_f1 is None or r["f1"] > best_f1["f1"]:
            best_f1 = {"threshold": float(t), **r}
        if best_min_fn is None or r["fn"] < best_min_fn["fn"] or (r["fn"] == best_min_fn["fn"] and r["f1"] > best_min_fn["f1"]):
            best_min_fn = {"threshold": float(t), **r}
    return best_f1, best_min_fn

print()
print("=" * 70)
print("F32 KERAS")
print("=" * 70)

probs_f32 = model.predict(test_inputs, batch_size=BATCH_SIZE, verbose=0).reshape(-1)
res_f32 = evaluate_predictions(y_test, probs_f32)

print(f"Accuracy : {res_f32['acc'] * 100:.2f}%")
print(f"Precision: {res_f32['prec'] * 100:.2f}%")
print(f"Recall   : {res_f32['rec'] * 100:.2f}%")
print(f"F1       : {res_f32['f1'] * 100:.2f}%")
print(f"FN       : {res_f32['fn']}")
print(f"FP       : {res_f32['fp']}")
print("Confusion Matrix:")
print(res_f32["cm"])

best_f1_f32, min_fn_f32 = search_thresholds(y_test, probs_f32)

print()
print("F32 threshold MAX F1")
print(f"Threshold: {best_f1_f32['threshold']:.2f}")
print(f"Accuracy : {best_f1_f32['acc'] * 100:.2f}%")
print(f"Precision: {best_f1_f32['prec'] * 100:.2f}%")
print(f"Recall   : {best_f1_f32['rec'] * 100:.2f}%")
print(f"F1       : {best_f1_f32['f1'] * 100:.2f}%")

print()
print("F32 threshold MIN FN")
print(f"Threshold: {min_fn_f32['threshold']:.2f}")
print(f"FN       : {min_fn_f32['fn']}")
print(f"FP       : {min_fn_f32['fp']}")

print()
print("=" * 70)
print("CONVERTING TFLITE F32")
print("=" * 70)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_f32 = converter.convert()
tflite_f32_path = os.path.join(OUT_DIR, "firegrid180_f32.tflite")

with open(tflite_f32_path, "wb") as f:
    f.write(tflite_f32)

print("Saved:", tflite_f32_path)

def predict_tflite(model_path, inputs):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = sorted(interpreter.get_input_details(), key=lambda d: d["name"])
    output_details = interpreter.get_output_details()
    n = len(inputs[0])
    probs = np.zeros(n, dtype=np.float32)
    for i in range(n):
        for j, detail in enumerate(input_details):
            tensor = np.expand_dims(inputs[j][i].astype(np.float32), axis=0)
            interpreter.set_tensor(detail["index"], tensor)
        interpreter.invoke()
        probs[i] = float(np.squeeze(interpreter.get_tensor(output_details[0]["index"])))
    return probs

probs_tflite_f32 = predict_tflite(tflite_f32_path, test_inputs)
res_tflite_f32 = evaluate_predictions(y_test, probs_tflite_f32)

print()
print("TFLITE F32")
print(f"Accuracy : {res_tflite_f32['acc'] * 100:.2f}%")
print(f"Precision: {res_tflite_f32['prec'] * 100:.2f}%")
print(f"Recall   : {res_tflite_f32['rec'] * 100:.2f}%")
print(f"F1       : {res_tflite_f32['f1'] * 100:.2f}%")
print(f"FN       : {res_tflite_f32['fn']}")
print(f"FP       : {res_tflite_f32['fp']}")

print()
print("=" * 70)
print("INT8 CALIBRATION")
print("=" * 70)

rng = np.random.default_rng(SEED)
idx_fire = np.where(y_train == 1)[0]
idx_nofire = np.where(y_train == 0)[0]
n_each = min(CALIB_N // 2, len(idx_fire), len(idx_nofire))
calib_fire = rng.choice(idx_fire, size=n_each, replace=False)
calib_nofire = rng.choice(idx_nofire, size=n_each, replace=False)
X_calib = np.concatenate([X_train[calib_fire], X_train[calib_nofire]], axis=0)
y_calib = np.concatenate([np.ones(n_each, dtype=np.int32), np.zeros(n_each, dtype=np.int32)])
perm = rng.permutation(len(X_calib))
X_calib = X_calib[perm]
y_calib = y_calib[perm]
calib_inputs = make_grid(X_calib)

print(f"Calibration total: {len(X_calib)}")
print(f"Calibration Fire  : {np.sum(y_calib == 1)}")
print(f"Calibration NoFire: {np.sum(y_calib == 0)}")

def representative_dataset():
    for i in range(len(calib_inputs[0])):
        yield [np.expand_dims(tile[i].astype(np.float32), axis=0) for tile in calib_inputs]

print()
print("=" * 70)
print("CONVERTING TFLITE INT8")
print("=" * 70)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32
tflite_int8 = converter.convert()
tflite_int8_path = os.path.join(OUT_DIR, "firegrid180_int8.tflite")

with open(tflite_int8_path, "wb") as f:
    f.write(tflite_int8)

print("Saved:", tflite_int8_path)

probs_int8 = predict_tflite(tflite_int8_path, test_inputs)
res_int8 = evaluate_predictions(y_test, probs_int8)

print()
print("=" * 70)
print("INT8 TFLITE")
print("=" * 70)

print(f"Accuracy : {res_int8['acc'] * 100:.2f}%")
print(f"Precision: {res_int8['prec'] * 100:.2f}%")
print(f"Recall   : {res_int8['rec'] * 100:.2f}%")
print(f"F1       : {res_int8['f1'] * 100:.2f}%")
print(f"FN       : {res_int8['fn']}")
print(f"FP       : {res_int8['fp']}")
print("Confusion Matrix:")
print(res_int8["cm"])

best_f1_int8, min_fn_int8 = search_thresholds(y_test, probs_int8)

print()
print("INT8 threshold MAX F1")
print(f"Threshold: {best_f1_int8['threshold']:.2f}")
print(f"Accuracy : {best_f1_int8['acc'] * 100:.2f}%")
print(f"Precision: {best_f1_int8['prec'] * 100:.2f}%")
print(f"Recall   : {best_f1_int8['rec'] * 100:.2f}%")
print(f"F1       : {best_f1_int8['f1'] * 100:.2f}%")

print()
print("INT8 threshold MIN FN")
print(f"Threshold: {min_fn_int8['threshold']:.2f}")
print(f"FN       : {min_fn_int8['fn']}")
print(f"FP       : {min_fn_int8['fp']}")

size_keras = os.path.getsize(os.path.join(OUT_DIR, "firegrid180_final.keras"))
size_f32 = os.path.getsize(tflite_f32_path)
size_int8 = os.path.getsize(tflite_int8_path)
params = model.count_params()

print()
print("=" * 70)
print("RESUMEN FINAL")
print("=" * 70)
print(f"Parameters       : {params:,}")
print(f"Keras size       : {size_keras / 1024:.2f} KB")
print(f"TFLite F32 size  : {size_f32 / 1024:.2f} KB")
print(f"TFLite INT8 size : {size_int8 / 1024:.2f} KB")
print()
print("                 F32(.keras)   TFLite(F32)   INT8")
print("-" * 70)
print(f"TEST ACC         {res_f32['acc']:.4f}          {res_tflite_f32['acc']:.4f}          {res_int8['acc']:.4f}")
print(f"TEST PREC        {res_f32['prec']:.4f}          {res_tflite_f32['prec']:.4f}          {res_int8['prec']:.4f}")
print(f"TEST REC         {res_f32['rec']:.4f}          {res_tflite_f32['rec']:.4f}          {res_int8['rec']:.4f}")
print(f"TEST F1          {res_f32['f1']:.4f}          {res_tflite_f32['f1']:.4f}          {res_int8['f1']:.4f}")
print(f"TEST FN          {res_f32['fn']:>8}          {res_tflite_f32['fn']:>8}          {res_int8['fn']:>8}")
print(f"TEST FP          {res_f32['fp']:>8}          {res_tflite_f32['fp']:>8}          {res_int8['fp']:>8}")
print()
print(f"INT8 threshold MAX F1 : {best_f1_int8['threshold']:.2f}")
print(f"INT8 threshold MIN FN : {min_fn_int8['threshold']:.2f}")
print()
print("Archivos guardados en:")
print(OUT_DIR)

# ---------------------------------------------------------------------
# EXTRA EXPERIMENT: compare grid sizes using the same train/val/test data
# The already-trained 4x4 F32 model supplies the 4x4 reference result.
# All metrics use the same held-out test set and threshold = 0.50.
# Other configurations are temporary and are not saved to disk.
# ---------------------------------------------------------------------
print()
print("=" * 70)
print("GRID SIZE COMPARISON (F32 KERAS)")
print("=" * 70)

comparison_results = {
    GRID: {
        "acc": float(res_f32["acc"]),
        "f1": float(res_f32["f1"]),
    }
}

# Release the main model's in-memory tensors before fitting temporary models.
del model, train_inputs, val_inputs, test_inputs
if "calib_inputs" in globals():
    del calib_inputs
if "X_calib" in globals():
    del X_calib
if "y_calib" in globals():
    del y_calib
if "converter" in globals():
    del converter
if "tflite_f32" in globals():
    del tflite_f32
if "tflite_int8" in globals():
    del tflite_int8
gc.collect()
tf.keras.backend.clear_session()

for grid in GRID_CONFIGS:
    if grid == GRID:
        print(f"Grid {grid}x{grid}: using the main model result.")
        continue

    print()
    print(f"Training comparison model: {grid}x{grid}")

    # Reset seeds for reproducible initialization/shuffling per configuration.
    np.random.seed(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)

    grid_train_inputs = make_grid(X_train_aug, grid=grid)
    grid_val_inputs = make_grid(X_val, grid=grid)
    grid_test_inputs = make_grid(X_test, grid=grid)

    grid_model = build_model(grid=grid)
    grid_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy"],
    )

    grid_callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.75, patience=10,
            min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15,
            restore_best_weights=True, verbose=1
        ),
    ]

    grid_model.fit(
        grid_train_inputs,
        y_train_aug,
        validation_data=(grid_val_inputs, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=grid_callbacks,
        verbose=1,
    )

    grid_probs = grid_model.predict(
        grid_test_inputs, batch_size=BATCH_SIZE, verbose=0
    ).reshape(-1)
    grid_metrics = evaluate_predictions(y_test, grid_probs, threshold=0.50)
    comparison_results[grid] = {
        "acc": float(grid_metrics["acc"]),
        "f1": float(grid_metrics["f1"]),
    }

    print(
        f"Grid {grid}x{grid} | Test Accuracy: {grid_metrics['acc'] * 100:.2f}% "
        f"| Test F1-score: {grid_metrics['f1'] * 100:.2f}%"
    )

    # Do not retain or save comparison models.
    del grid_model, grid_train_inputs, grid_val_inputs, grid_test_inputs
    del grid_probs, grid_metrics, grid_callbacks
    gc.collect()
    tf.keras.backend.clear_session()

# Sort all six configurations before plotting.
plot_grids = sorted(comparison_results.keys())
plot_labels = [f"{grid}x{grid}" for grid in plot_grids]
f1_values = [comparison_results[grid]["f1"] * 100.0 for grid in plot_grids]
acc_values = [comparison_results[grid]["acc"] * 100.0 for grid in plot_grids]

# Publication-style figures: clean typography, restrained colors, subtle grid,
# percent annotations, and 600-dpi raster export for paper submission.
def save_metric_plot(values, y_label, filename, color):
    """Create a paper-ready comparison plot with axes only (no title/annotations)."""
    x = np.arange(len(plot_grids))
    values = np.asarray(values, dtype=float)

    with plt.rc_context({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }):
        fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
        ax.plot(
            x, values,
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=6.0,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.6,
            zorder=3,
        )

        spread = float(np.ptp(values))
        pad = max(spread * 0.30, 0.45)
        y_bottom = max(0.0, float(np.floor(np.min(values) - pad - 0.25)))
        y_top = min(100.0, float(np.ceil(np.max(values) + pad + 0.5)))
        if y_top - y_bottom < 1.5:
            y_bottom = max(0.0, y_top - 1.5)
        ax.set_ylim(y_bottom, y_top)

        # Keep only axis labels and tick labels; the paper caption identifies the figure.
        ax.set_xlabel("Grid configuration (rows × columns)", labelpad=8, color="#303846")
        ax.set_ylabel(y_label, labelpad=8, color="#303846")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{grid} × {grid}" for grid in plot_grids])
        ax.grid(axis="y", color="#DDE3EA", linewidth=0.75, linestyle="--", zorder=0)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="both", which="major", length=0, pad=7, colors="#465366")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#C7CFD9")
        ax.spines["bottom"].set_color("#C7CFD9")
        ax.margins(x=0.08)

        fig.savefig(
            os.path.join(OUT_DIR, filename),
            dpi=600,
            bbox_inches="tight",
            metadata={"Software": "Matplotlib"},
        )
        plt.close(fig)

# Figure 1: F1-score comparison.
save_metric_plot(
    f1_values,
    y_label="F1-Score (%)",
    filename="grid_comparison_f1.png",
    color="#2457A7",
)

# Figure 2: accuracy comparison.
save_metric_plot(
    acc_values,
    y_label="Accuracy (%)",
    filename="grid_comparison_accuracy.png",
    color="#187F78",
)

print()
print("GRID COMPARISON SUMMARY (TEST SET)")
print("Grid       Accuracy (%)       F1-Score (%)")
print("-------------------------------------------")
for grid in plot_grids:
    print(
        f"{grid}x{grid:<5}  "
        f"{comparison_results[grid]['acc'] * 100:>10.2f}          "
        f"{comparison_results[grid]['f1'] * 100:>10.2f}"
    )
print()
print("Additional plots saved:")
print(os.path.join(OUT_DIR, "grid_comparison_f1.png"))
print(os.path.join(OUT_DIR, "grid_comparison_accuracy.png"))

# ---------------------------------------------------------------------
# EXTRA EXPERIMENT: multi-seed robustness comparison
# Seed 42 reuses the already-computed official grid results.
# Four additional seeds are generated randomly for each script execution.
# Only the two aggregate plots are saved; temporary models stay in RAM.
# ---------------------------------------------------------------------
SEED_COUNT = 5
random_seed_source = random.SystemRandom()
additional_seeds = []
while len(additional_seeds) < SEED_COUNT - 1:
    candidate_seed = random_seed_source.randrange(1, 2**31 - 1)
    if candidate_seed != SEED and candidate_seed not in additional_seeds:
        additional_seeds.append(candidate_seed)

SEED_SWEEP = [SEED] + additional_seeds

seed_results = {
    SEED: {
        int(grid): {
            "acc": float(metrics["acc"]),
            "f1": float(metrics["f1"]),
        }
        for grid, metrics in comparison_results.items()
    }
}

print()
print("=" * 70)
print("MULTI-SEED GRID COMPARISON")
print("=" * 70)
print("Seeds used (seed 42 is the official baseline):", SEED_SWEEP)
print("The train/validation split and held-out test set remain unchanged.")
print("Additional models are temporary and are not saved to disk.")

for experiment_seed in additional_seeds:
    print()
    print(f"Running additional seed: {experiment_seed}")

    # Each seed controls augmentation, initialization, dropout and shuffling.
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)
    tf.random.set_seed(experiment_seed)
    seed_X_train_aug, seed_y_train_aug = augment_dataset(X_train, y_train)

    seed_results[experiment_seed] = {}

    for grid in GRID_CONFIGS:
        np.random.seed(experiment_seed)
        random.seed(experiment_seed)
        tf.random.set_seed(experiment_seed)

        grid_seed_train_inputs = make_grid(seed_X_train_aug, grid=grid)
        grid_seed_val_inputs = make_grid(X_val, grid=grid)
        grid_seed_test_inputs = make_grid(X_test, grid=grid)

        seed_model = build_model(grid=grid)
        seed_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
            loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING),
            metrics=["accuracy"],
        )

        seed_callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.75, patience=10,
                min_lr=1e-6, verbose=0
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=15,
                restore_best_weights=True, verbose=0
            ),
        ]

        seed_model.fit(
            grid_seed_train_inputs,
            seed_y_train_aug,
            validation_data=(grid_seed_val_inputs, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weight,
            callbacks=seed_callbacks,
            verbose=0,
        )

        seed_probs = seed_model.predict(
            grid_seed_test_inputs, batch_size=BATCH_SIZE, verbose=0
        ).reshape(-1)
        seed_metrics = evaluate_predictions(y_test, seed_probs, threshold=0.50)
        seed_results[experiment_seed][grid] = {
            "acc": float(seed_metrics["acc"]),
            "f1": float(seed_metrics["f1"]),
        }

        print(
            f"Seed {experiment_seed} | Grid {grid}x{grid} | "
            f"Accuracy: {seed_metrics['acc'] * 100:.2f}% | "
            f"F1-score: {seed_metrics['f1'] * 100:.2f}%"
        )

        # Do not save or retain any additional model.
        del (
            seed_model, grid_seed_train_inputs, grid_seed_val_inputs,
            grid_seed_test_inputs, seed_probs, seed_metrics, seed_callbacks
        )
        gc.collect()
        tf.keras.backend.clear_session()

    del seed_X_train_aug, seed_y_train_aug
    gc.collect()

# Aggregate across all five seeds: mean with standard-deviation error bars.
seed_plot_grids = sorted(GRID_CONFIGS)


def save_seed_mean_plot(metric_key, y_label, filename, color):
    means = []
    stds = []
    for grid in seed_plot_grids:
        values = np.array(
            [seed_results[seed][grid][metric_key] * 100.0 for seed in SEED_SWEEP],
            dtype=float,
        )
        means.append(float(np.mean(values)))
        stds.append(float(np.std(values, ddof=1)))

    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)
    x = np.arange(len(seed_plot_grids))

    with plt.rc_context({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }):
        fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
        ax.errorbar(
            x,
            means,
            yerr=stds,
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=6.0,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.5,
            capsize=4,
            elinewidth=1.2,
            ecolor="#667085",
            zorder=3,
        )

        # Annotate mean percentages for quick comparison.
        for xi, mean_value, std_value in zip(x, means, stds):
            ax.annotate(
                f"{mean_value:.2f}%",
                xy=(xi, mean_value + std_value),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color=color,
            )

        ax.set_ylim(0, 100)
        ax.set_yticks(np.arange(0, 101, 10))
        ax.set_xlabel("Grid configuration (rows × columns)", labelpad=8, color="#303846")
        ax.set_ylabel(y_label, labelpad=8, color="#303846")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{grid} × {grid}" for grid in seed_plot_grids])
        ax.grid(axis="y", color="#DDE3EA", linewidth=0.75, linestyle="--", zorder=0)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="both", which="major", length=0, pad=7, colors="#465366")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#C7CFD9")
        ax.spines["bottom"].set_color("#C7CFD9")
        ax.margins(x=0.08)
        fig.savefig(
            os.path.join(OUT_DIR, filename),
            dpi=600,
            bbox_inches="tight",
            metadata={"Software": "Matplotlib"},
        )
        plt.close(fig)


save_seed_mean_plot(
    metric_key="f1",
    y_label="Mean F1-Score (%)",
    filename="grid_seed_comparison_f1.png",
    color="#2457A7",
)
save_seed_mean_plot(
    metric_key="acc",
    y_label="Mean Accuracy (%)",
    filename="grid_seed_comparison_accuracy.png",
    color="#187F78",
)

print()
print("MULTI-SEED SUMMARY (TEST SET; MEAN ± STANDARD DEVIATION)")
print("Grid       Accuracy (%)             F1-Score (%)")
print("-------------------------------------------------")
for grid in seed_plot_grids:
    acc_values = np.array(
        [seed_results[seed][grid]["acc"] * 100.0 for seed in SEED_SWEEP]
    )
    f1_values = np.array(
        [seed_results[seed][grid]["f1"] * 100.0 for seed in SEED_SWEEP]
    )
    print(
        f"{grid}x{grid:<5}  "
        f"{np.mean(acc_values):>6.2f} ± {np.std(acc_values, ddof=1):<6.2f}      "
        f"{np.mean(f1_values):>6.2f} ± {np.std(f1_values, ddof=1):<6.2f}"
    )

print()
print("Multi-seed plots saved:")
print(os.path.join(OUT_DIR, "grid_seed_comparison_f1.png"))
print(os.path.join(OUT_DIR, "grid_seed_comparison_accuracy.png"))

# Final cleanup. Original model files and original metrics remain unchanged.
del X_train, X_val, X_test, X_train_aug, y_train, y_val, y_test, y_train_aug
if "comparison_results" in globals():
    del comparison_results
gc.collect()

    )

print()
print("Multi-seed plots saved:")
print(os.path.join(OUT_DIR, "grid_seed_comparison_f1.png"))
print(os.path.join(OUT_DIR, "grid_seed_comparison_accuracy.png"))

# Final cleanup. Original model files and original metrics remain unchanged.
del X_train, X_val, X_test, X_train_aug, y_train, y_val, y_test, y_train_aug
if "comparison_results" in globals():
    del comparison_results
gc.collect()

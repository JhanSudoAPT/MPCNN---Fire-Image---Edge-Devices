
import os
import json
import gc
import numpy as np
import cv2
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

SEED = 42
IMG_SIZE = 180
GRID = 4
TILE_SIZE = IMG_SIZE // GRID
CALIB_N = 400
CALIB_BATCH = 8
TEST_BATCH = 8

TRAIN_DIR = r"PATH_TO_TRAIN_DATASET"
TEST_DIR = r"PATH_TO_TEST_DATASET"
OUT_DIR = r"PATH_TO_OUTPUT_DIRECTORY"

MODEL_CANDIDATES = [
    os.path.join(OUT_DIR, "firegrid.keras"),
    os.path.join(OUT_DIR, "firegrid180_final.keras"),
    os.path.join(OUT_DIR, "firegrid180_best.keras"),
]

REPORT_PATH = os.path.join(
    OUT_DIR, "MPCNN_INT8_DIAGNOSTIC.json"
)

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

np.random.seed(SEED)
tf.random.set_seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)


def get_images(folder):
    files = []
    for root, _, names in os.walk(folder):
        for name in names:
            if os.path.splitext(name)[1].lower() in EXTS:
                files.append(os.path.join(root, name))
    return sorted(files)


def list_labeled_images(base_dir):
    items = []

    for label, cls in enumerate(("NoFire", "Fire")):
        folder = os.path.join(base_dir, cls)

        if not os.path.isdir(folder):
            raise FileNotFoundError(
                f"Class directory does not exist: {folder}"
            )

        files = get_images(folder)
        print(f"{cls:8s}: {len(files)}")
        items.extend((path, label) for path in files)

    return items


def read_rgb_u8(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(
        img,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_AREA
    )

    return img


def load_selected(items):
    images, labels, valid_paths = [], [], []

    for path, label in items:
        img = read_rgb_u8(path)

        if img is None:
            continue

        images.append(img)
        labels.append(label)
        valid_paths.append(path)

    if not images:
        return (
            np.empty((0, IMG_SIZE, IMG_SIZE, 3), np.uint8),
            np.empty((0,), np.int32),
            []
        )

    return (
        np.asarray(images, dtype=np.uint8),
        np.asarray(labels, dtype=np.int32),
        valid_paths
    )


def make_grid_float(images):
    tiles = []

    for gy in range(GRID):
        for gx in range(GRID):
            y0 = gy * TILE_SIZE
            y1 = y0 + TILE_SIZE
            x0 = gx * TILE_SIZE
            x1 = x0 + TILE_SIZE

            tiles.append(images[:, y0:y1, x0:x1, :])

    return tiles


def choose_calibration_items(train_items):
    rng = np.random.default_rng(SEED)
    by_class = {0: [], 1: []}

    for item in train_items:
        by_class[item[1]].append(item)

    n_each = min(
        CALIB_N // 2,
        len(by_class[0]),
        len(by_class[1])
    )

    selected = []

    for cls in (0, 1):
        indices = rng.choice(
            len(by_class[cls]),
            size=n_each,
            replace=False
        )
        selected.extend(
            by_class[cls][int(i)] for i in indices
        )

    rng.shuffle(selected)
    return selected


def positive_scale(max_value):
    return max(float(max_value) / 255.0, 1e-8)


def collect_activation_scales(model, calibration_items):
    tile_model = model.get_layer("shared_tile_cnn")

    tile_layer_names = [
        "conv1d_1",
        "conv1d_2",
        "conv1d_3",
        "gap",
        "tile_dense32"
    ]

    tile_tap = tf.keras.Model(
        tile_model.inputs,
        [
            tile_model.get_layer(name).output
            for name in tile_layer_names
        ]
    )

    head_tap = tf.keras.Model(
        model.inputs,
        [
            model.get_layer("concat_tiles").output,
            model.get_layer("mlp_dense16").output
        ]
    )

    maxima = {name: 0.0 for name in tile_layer_names}
    maxima["concat_tiles"] = 0.0
    maxima["mlp_dense16"] = 0.0

    print("\nCalibrating activations using real images...")

    processed = 0

    for start in range(0, len(calibration_items), CALIB_BATCH):
        batch_items = calibration_items[
            start:start + CALIB_BATCH
        ]

        xu8, _, _ = load_selected(batch_items)

        if len(xu8) == 0:
            continue

        xf = xu8.astype(np.float32) / 255.0

        patch_batches = make_grid_float(xf)
        all_tiles = np.concatenate(patch_batches, axis=0)

        tile_acts = tile_tap(
            {"tile_input": all_tiles},
            training=False
        )

        for name, values in zip(tile_layer_names, tile_acts):
            maxima[name] = max(
                maxima[name],
                float(tf.reduce_max(values).numpy())
            )

        concat_act, mlp_act = head_tap(
            patch_batches,
            training=False
        )

        maxima["concat_tiles"] = max(
            maxima["concat_tiles"],
            float(tf.reduce_max(concat_act).numpy())
        )

        maxima["mlp_dense16"] = max(
            maxima["mlp_dense16"],
            float(tf.reduce_max(mlp_act).numpy())
        )

        processed += len(xu8)
        print(
            f"  Calibration: {processed}/"
            f"{len(calibration_items)}",
            end="\r"
        )

    print()

    scales = {
        "input": 1.0 / 255.0,
        "conv1": positive_scale(maxima["conv1d_1"]),
        "conv2": positive_scale(maxima["conv1d_2"]),
        "conv3": positive_scale(maxima["conv1d_3"]),
        "gap": positive_scale(maxima["gap"]),
        "tile_dense": positive_scale(
            max(maxima["tile_dense32"], maxima["concat_tiles"])
        ),
        "mlp_dense": positive_scale(maxima["mlp_dense16"])
    }

    print("\nCalibrated scales:")
    for name, value in scales.items():
        print(f"  {name:12s}: {value:.10g}")

    return scales, maxima


def quantize_weights_per_output_channel(weights, bias, input_scale):
    weights = np.asarray(weights, dtype=np.float32)
    bias = np.asarray(bias, dtype=np.float32)

    axes = tuple(range(weights.ndim - 1))
    max_abs = np.max(np.abs(weights), axis=axes)
    weight_scales = np.maximum(
        max_abs / 127.0, 1e-12
    ).astype(np.float32)

    shape = (1,) * (weights.ndim - 1) + (weights.shape[-1],)

    quantized_weights = np.clip(
        np.rint(weights / weight_scales.reshape(shape)),
        -127,
        127
    ).astype(np.int8)

    bias_scales = (
        np.float64(input_scale) *
        weight_scales.astype(np.float64)
    )

    quantized_bias = np.clip(
        np.rint(
            bias.astype(np.float64) / bias_scales
        ),
        -2147483648,
        2147483647
    ).astype(np.int32)

    return quantized_weights, quantized_bias, weight_scales


def build_quant_params(model, scales):
    tile_model = model.get_layer("shared_tile_cnn")

    specifications = [
        (
            "conv1", tile_model.get_layer("conv1d_1"),
            scales["input"], scales["conv1"]
        ),
        (
            "conv2", tile_model.get_layer("conv1d_2"),
            scales["conv1"], scales["conv2"]
        ),
        (
            "conv3", tile_model.get_layer("conv1d_3"),
            scales["conv2"], scales["conv3"]
        ),
        (
            "tile_dense", tile_model.get_layer("tile_dense32"),
            scales["gap"], scales["tile_dense"]
        ),
        (
            "mlp_dense", model.get_layer("mlp_dense16"),
            scales["tile_dense"], scales["mlp_dense"]
        ),
        (
            "output", model.get_layer("output"),
            scales["mlp_dense"], None
        )
    ]

    params = {}

    for name, layer, input_scale, output_scale in specifications:
        weights, bias = layer.get_weights()

        qw, qb, weight_scales = quantize_weights_per_output_channel(
            weights, bias, input_scale
        )

        entry = {
            "weights": qw,
            "bias": qb,
            "weight_scales": weight_scales,
            "input_scale": float(input_scale)
        }

        if output_scale is not None:
            entry["multiplier"] = (
                np.float32(input_scale) * weight_scales /
                np.float32(output_scale)
            ).astype(np.float32)
        else:
            entry["logit_scale"] = (
                np.float32(input_scale) * weight_scales[0]
            )

        params[name] = entry

    return params


def requantize_relu(acc, multiplier):
    scaled = (
        np.asarray(acc, dtype=np.float32) *
        np.asarray(multiplier, dtype=np.float32)
    )

    levels = np.floor(
        np.maximum(scaled, 0.0) + np.float32(0.5)
    )

    return np.clip(
        levels - 128.0, -128, 127
    ).astype(np.int8)


def conv1d_first(input_u8, weights, bias, multiplier):
    length, _ = input_u8.shape
    kernel, _, output_channels = weights.shape
    pad_left = (kernel - 1) // 2
    pad_right = kernel - 1 - pad_left

    padded = np.pad(
        input_u8.astype(np.int16),
        ((pad_left, pad_right), (0, 0)),
        mode="constant",
        constant_values=0
    )

    windows = np.stack(
        [padded[k:k + length] for k in range(kernel)],
        axis=1
    )

    acc = np.einsum(
        "tki,kio->to",
        windows.astype(np.int32),
        weights.astype(np.int32),
        dtype=np.int32,
        optimize=True
    )

    acc += bias.reshape(1, output_channels)

    return requantize_relu(
        acc, multiplier.reshape(1, output_channels)
    )


def conv1d_int8(input_q, weights, bias, multiplier):
    length, _ = input_q.shape
    kernel, _, output_channels = weights.shape
    pad_left = (kernel - 1) // 2
    pad_right = kernel - 1 - pad_left

    # Asymmetric signed activation representation:
    # real = (q + 128) * scale.
    centered = input_q.astype(np.int16) + 128

    # Padding represents real zero, hence centered value zero.
    padded = np.pad(
        centered,
        ((pad_left, pad_right), (0, 0)),
        mode="constant",
        constant_values=0
    )

    windows = np.stack(
        [padded[k:k + length] for k in range(kernel)],
        axis=1
    )

    acc = np.einsum(
        "tki,kio->to",
        windows.astype(np.int32),
        weights.astype(np.int32),
        dtype=np.int32,
        optimize=True
    )

    acc += bias.reshape(1, output_channels)

    return requantize_relu(
        acc, multiplier.reshape(1, output_channels)
    )


def max_pool2_q(input_q):
    length, channels = input_q.shape
    output_length = length // 2

    return input_q[:output_length * 2].reshape(
        output_length, 2, channels
    ).max(axis=1).astype(np.int8)


def global_average_pool_q(input_q, input_scale, output_scale):
    length, _ = input_q.shape

    centered_sums = (
        input_q.astype(np.int32) + 128
    ).sum(axis=0, dtype=np.int32)

    mean_real = (
        centered_sums.astype(np.float64) / length
    ) * float(input_scale)

    levels = np.floor(
        mean_real / float(output_scale) + 0.5
    )

    return np.clip(
        levels - 128.0, -128, 127
    ).astype(np.int8)


def dense_relu_q(input_q, weights, bias, multiplier):
    centered = input_q.astype(np.int16) + 128

    acc = np.einsum(
        "i,io->o",
        centered.astype(np.int32),
        weights.astype(np.int32),
        dtype=np.int32,
        optimize=True
    )

    acc += bias

    return requantize_relu(acc, multiplier)


def manual_predict_one(image_u8, params, scales):
    features = np.empty(
        (GRID * GRID * 32,), dtype=np.int8
    )

    p1 = params["conv1"]
    p2 = params["conv2"]
    p3 = params["conv3"]
    pd = params["tile_dense"]
    ph = params["mlp_dense"]
    po = params["output"]

    for gy in range(GRID):
        for gx in range(GRID):
            y0 = gy * TILE_SIZE
            y1 = y0 + TILE_SIZE
            x0 = gx * TILE_SIZE
            x1 = x0 + TILE_SIZE

            patch = image_u8[y0:y1, x0:x1, :]
            rows = patch.reshape(TILE_SIZE, TILE_SIZE * 3)

            a1 = conv1d_first(
                rows, p1["weights"], p1["bias"],
                p1["multiplier"]
            )
            a1 = max_pool2_q(a1)

            a2 = conv1d_int8(
                a1, p2["weights"], p2["bias"],
                p2["multiplier"]
            )
            a2 = max_pool2_q(a2)

            a3 = conv1d_int8(
                a2, p3["weights"], p3["bias"],
                p3["multiplier"]
            )

            gap = global_average_pool_q(
                a3, scales["conv3"], scales["gap"]
            )

            tile_features = dense_relu_q(
                gap, pd["weights"], pd["bias"],
                pd["multiplier"]
            )

            tile_index = gy * GRID + gx
            start = tile_index * 32
            features[start:start + 32] = tile_features

    hidden = dense_relu_q(
        features, ph["weights"], ph["bias"],
        ph["multiplier"]
    )

    centered = hidden.astype(np.int16) + 128

    acc = int(np.dot(
        centered.astype(np.int64),
        po["weights"].reshape(-1).astype(np.int64)
    ))

    acc += int(po["bias"][0])

    logit = float(
        np.float64(acc) * float(po["logit_scale"])
    )

    if logit >= 0.0:
        return float(1.0 / (1.0 + np.exp(-logit)))

    e = float(np.exp(logit))
    return float(e / (1.0 + e))


def debug_manual_quantization(model, image_u8, params, scales):
    tile_model = model.get_layer("shared_tile_cnn")

    layer_names = [
        "conv1d_1",
        "pool1d_1",
        "conv1d_2",
        "pool1d_2",
        "conv1d_3",
        "gap",
        "tile_dense32"
    ]

    tap = tf.keras.Model(
        tile_model.inputs,
        [
            tile_model.get_layer(name).output
            for name in layer_names
        ]
    )

    patch = image_u8[:TILE_SIZE, :TILE_SIZE, :]
    xf = patch.astype(np.float32) / 255.0

    refs = tap(
        {"tile_input": xf[None, ...]},
        training=False
    )
    refs = [np.asarray(v)[0] for v in refs]

    rows = patch.reshape(TILE_SIZE, TILE_SIZE * 3)

    p1 = params["conv1"]
    p2 = params["conv2"]
    p3 = params["conv3"]
    pd = params["tile_dense"]

    q1 = conv1d_first(
        rows, p1["weights"], p1["bias"],
        p1["multiplier"]
    )
    qp1 = max_pool2_q(q1)

    q2 = conv1d_int8(
        qp1, p2["weights"], p2["bias"],
        p2["multiplier"]
    )
    qp2 = max_pool2_q(q2)

    q3 = conv1d_int8(
        qp2, p3["weights"], p3["bias"],
        p3["multiplier"]
    )

    qgap = global_average_pool_q(
        q3, scales["conv3"], scales["gap"]
    )

    qdense = dense_relu_q(
        qgap, pd["weights"], pd["bias"],
        pd["multiplier"]
    )

    quantized = [q1, qp1, q2, qp2, q3, qgap, qdense]

    activation_scales = [
        scales["conv1"],
        scales["conv1"],
        scales["conv2"],
        scales["conv2"],
        scales["conv3"],
        scales["gap"],
        scales["tile_dense"]
    ]

    print("\n" + "=" * 76)
    print("LAYER-BY-LAYER DIAGNOSTICS: KERAS F32 vs MANUAL INT8")
    print("Reconstruction: real = (q + 128) * scale")
    print("=" * 76)

    results = []

    for name, ref, q, scale in zip(
        layer_names, refs, quantized, activation_scales
    ):
        dequant = (q.astype(np.float32) + 128.0) * scale
        error = np.abs(ref - dequant)

        result = {
            "layer": name,
            "mae": float(np.mean(error)),
            "max_error": float(np.max(error)),
            "keras_max": float(np.max(ref)),
            "int8_dequant_max": float(np.max(dequant)),
            "saturated_high_pct": float(np.mean(q == 127) * 100),
            "saturated_low_pct": float(np.mean(q == -128) * 100)
        }

        results.append(result)

        print(
            f"{name:14s} "
            f"MAE={result['mae']:.6f}  "
            f"MAX_ERR={result['max_error']:.6f}  "
            f"F32_MAX={result['keras_max']:.5f}  "
            f"INT8_MAX={result['int8_dequant_max']:.5f}  "
            f"SAT+={result['saturated_high_pct']:.2f}%  "
            f"SAT-={result['saturated_low_pct']:.2f}%"
        )

    return results


def eval_metrics(y_true, probs):
    pred = (np.asarray(probs) >= 0.5).astype(np.int32)

    cm = confusion_matrix(
        y_true, pred, labels=[0, 1]
    )

    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(
            precision_score(y_true, pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, pred, zero_division=0)
        ),
        "fn": int(cm[1, 0]),
        "fp": int(cm[0, 1]),
        "confusion_matrix": cm.tolist()
    }


def print_metrics(title, result):
    print(f"\n{title}")
    print(
        f"ACC: {result['accuracy'] * 100:.2f}% | "
        f"PREC: {result['precision'] * 100:.2f}% | "
        f"REC: {result['recall'] * 100:.2f}% | "
        f"F1: {result['f1'] * 100:.2f}%"
    )
    print(
        f"FN: {result['fn']} | FP: {result['fp']} | "
        f"CM: {result['confusion_matrix']}"
    )


def main():
    print("=" * 76)
    print("MPCNN - PTQ INT8 DIAGNOSTICS")
    print("Does not export or overwrite MPCNN_weights.h")
    print("=" * 76)

    model_path = next(
        (p for p in MODEL_CANDIDATES if os.path.isfile(p)),
        None
    )

    if model_path is None:
        available = [
            p for p in os.listdir(OUT_DIR)
            if p.lower().endswith((".keras", ".h5"))
        ]

        if len(available) == 1:
            model_path = os.path.join(OUT_DIR, available[0])
        else:
            raise FileNotFoundError(
                "A unique Keras model could not be found in "
                + OUT_DIR
                + f"\nAvailable models: {available}"
            )

    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(f"TRAIN_DIR does not exist: {TRAIN_DIR}")

    if not os.path.isdir(TEST_DIR):
        raise FileNotFoundError(f"TEST_DIR does not exist: {TEST_DIR}")

    print("Model:", model_path)
    print("TRAIN:", TRAIN_DIR)
    print("TEST :", TEST_DIR)

    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    if len(model.inputs) != GRID * GRID:
        raise ValueError(
            f"Expected 16 inputs; the model has {len(model.inputs)}."
        )

    tile_model = model.get_layer("shared_tile_cnn")

    if tuple(tile_model.input_shape[1:]) != (
        TILE_SIZE, TILE_SIZE, 3
    ):
        raise ValueError(
            f"Unexpected tile input shape: {tile_model.input_shape}"
        )

    print("Parameters:", f"{model.count_params():,}")

    print("\nSearching for calibration images...")
    train_items = list_labeled_images(TRAIN_DIR)
    calibration_items = choose_calibration_items(train_items)

    print(f"\nTotal calibration images: {len(calibration_items)}")

    scales, maxima = collect_activation_scales(
        model, calibration_items
    )
    params = build_quant_params(model, scales)

    # Diagnose the first available image in the TEST set.
    test_items = list_labeled_images(TEST_DIR)

    if not test_items:
        raise RuntimeError("No test images were found.")

    debug_image = None

    for path, _ in test_items:
        debug_image = read_rgb_u8(path)
        if debug_image is not None:
            break

    if debug_image is None:
        raise RuntimeError("Could not read a test image.")

    layer_diagnostics = debug_manual_quantization(
        model, debug_image, params, scales
    )

    print("\nEvaluating the TEST set...")

    y_all = []
    probs_f32 = []
    probs_manual = []

    for start in range(0, len(test_items), TEST_BATCH):
        items = test_items[start:start + TEST_BATCH]
        xu8, labels, _ = load_selected(items)

        if len(xu8) == 0:
            continue

        xf = xu8.astype(np.float32) / 255.0
        inputs = make_grid_float(xf)

        pred_f32 = np.asarray(
            model(inputs, training=False)
        ).reshape(-1)

        pred_manual = np.asarray(
            [
                manual_predict_one(image, params, scales)
                for image in xu8
            ],
            dtype=np.float32
        )

        y_all.extend(labels.tolist())
        probs_f32.extend(pred_f32.tolist())
        probs_manual.extend(pred_manual.tolist())

        print(
            f"  Test: {len(y_all)}/{len(test_items)}",
            end="\r"
        )

    print()

    y_all = np.asarray(y_all, dtype=np.int32)
    probs_f32 = np.asarray(probs_f32, dtype=np.float32)
    probs_manual = np.asarray(probs_manual, dtype=np.float32)

    if len(y_all) == 0:
        raise RuntimeError("No images could be evaluated.")

    results_f32 = eval_metrics(y_all, probs_f32)
    results_manual = eval_metrics(y_all, probs_manual)

    mae = float(np.mean(np.abs(probs_f32 - probs_manual)))
    max_error = float(np.max(np.abs(probs_f32 - probs_manual)))

    print("\n" + "=" * 76)
    print_metrics("KERAS F32", results_f32)
    print_metrics("MANUAL PTQ INT8 - NumPy", results_manual)

    print(
        f"\nProbability error: MAE={mae:.6f} | "
        f"MAX={max_error:.6f}"
    )
    print("=" * 76)

    report = {
        "model_path": model_path,
        "image_size": IMG_SIZE,
        "grid": GRID,
        "tile_size": TILE_SIZE,
        "params": int(model.count_params()),
        "calibration_count": len(calibration_items),
        "activation_maxima": maxima,
        "activation_scales": scales,
        "layer_diagnostics": layer_diagnostics,
        "test_count": len(y_all),
        "keras_f32": results_f32,
        "manual_ptq_int8": results_manual,
        "probability_mae": mae,
        "probability_max_abs_error": max_error,
        "header_exported": False
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\nReport:", REPORT_PATH)
    print("MPCNN_weights.h has not been modified.")

    del model
    gc.collect()


if __name__ == "__main__":
    main()

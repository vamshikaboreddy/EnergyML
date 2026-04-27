# ===========================================================
# train.py — Data, Models, and Training
# Energy-Aware ML Project with EfficientNetV2M
# Dataset: Oxford Flowers102 (via tensorflow_datasets)
#
# Techniques:
#   1. Baseline        — EfficientNetV2M, 2-phase (head → fine-tune)
#   2. Quantization    — Float16 TFLite
#   3. Early Stopping  — Auto-stop on val_accuracy plateau
#   4. Transfer Learning — Head-only + partial unfreeze (2-phase)
#   5. Fine-Tuning     — Deep layer unfreeze with cosine warmup LR
# ===========================================================

print("=" * 70)
print("  ENERGY-AWARE MACHINE LEARNING — EfficientNetV2M + Flowers102")
print("=" * 70)
print("\n[1] Loading libraries...")

import os
import gc
import json
import time
import warnings
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from codecarbon import EmissionsTracker

try:
    import tf_keras as keras
    from tf_keras import layers
except ImportError:
    from tensorflow import keras
    from tensorflow.keras import layers

warnings.filterwarnings('ignore')

from config import *

np.random.seed(SEED)
tf.random.set_seed(SEED)

print(f"✅ TensorFlow: {tf.__version__}")
print(f"✅ Model: {MODEL_NAME} | Pretrained: {USE_PRETRAINED}")

for d in [RESULTS_DIR, MODELS_DIR, PLOTS_DIR, DATA_DIR]:
    os.makedirs(d, exist_ok=True)
print("✅ Output folders ready")


# ===========================================================
# SECTION 1: DATA LOADING — Oxford Flowers102
# ===========================================================

print("\n" + "=" * 70)
print("[2] Loading Oxford Flowers102 Dataset (via tensorflow_datasets)")
print("=" * 70)
print("  First run will download ~330 MB — subsequent runs use cache.")

(ds_train_raw, ds_val_raw, ds_test_raw), ds_info = tfds.load(
    DATASET_NAME,
    split        = ['train', 'validation', 'test'],
    as_supervised= True,
    with_info    = True
)

def preprocess(image, label):
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image, tf.float32)
    return image, label

ds_train_proc = ds_test_raw.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).cache()
ds_test_proc  = ds_train_raw.concatenate(ds_val_raw)
ds_test_proc  = ds_test_proc.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).cache()

print("  Converting test set to numpy arrays...")

def dataset_to_numpy(ds):
    images, labels = [], []
    for img, lbl in ds:
        images.append(img.numpy())
        labels.append(lbl.numpy())
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)

x_test, y_test = dataset_to_numpy(ds_test_proc)
TRAIN_SIZE     = sum(1 for _ in ds_train_proc)
TEST_SIZE      = len(x_test)

print(f"✅ Training images : {TRAIN_SIZE}")
print(f"✅ Test images     : {TEST_SIZE}")
print(f"✅ Image shape     : ({IMAGE_SIZE}, {IMAGE_SIZE}, 3)")
print(f"✅ Classes         : {NUM_CLASSES}  (102 flower species)")
print(f"✅ Label range     : {y_test.min()} – {y_test.max()}")


def make_train_dataset(batch_size, shuffle=True):
    ds = ds_train_proc
    if shuffle:
        ds = ds.shuffle(buffer_size=2048, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def make_test_dataset(batch_size):
    return ds_test_proc.batch(batch_size).prefetch(tf.data.AUTOTUNE)


data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomContrast(0.1),
], name="data_augmentation")

strong_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.2),
    layers.RandomTranslation(0.15, 0.15),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),
], name="strong_augmentation")


# ===========================================================
# SECTION 2: HELPER UTILITIES
# ===========================================================

def save_result(name, result):
    path = os.path.join(DATA_DIR, f"result_{name.replace(' ', '_')}.json")
    data = {
        'accuracy' : float(result.get('accuracy', 0.0)),
        'size'     : float(result.get('size', 0.0)),
        'params'   : str(result.get('params', 'N/A')),
        'co2'      : float(result.get('co2', 0.0)),
        'time'     : float(result.get('time', 0.0)),
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  💾 Metrics saved → {path}")


def measure_energy(func, name):
    tracker = EmissionsTracker(
        output_dir=DATA_DIR,
        save_to_file=False,
        project_name=name,
        log_level='error'
    )
    tracker.start()
    t0 = time.time()
    try:
        result = func()
    finally:
        elapsed   = time.time() - t0
        emissions = tracker.stop() or 0.0
    print(f"  ⚡ [{name}] Time: {elapsed:.1f}s  |  CO₂: {emissions:.6f} kg")
    return result, emissions, elapsed


def get_model_size_mb(model):
    if isinstance(model, bytes):
        return len(model) / (1024 ** 2)
    if hasattr(model, 'count_params'):
        return model.count_params() * 4 / (1024 ** 2)
    return 0.0


def evaluate_tflite(tflite_model_bytes, x, y, n_samples=None):
    n_samples = min(n_samples or QUANT_TEST_SAMPLES, len(x))
    interpreter = tf.lite.Interpreter(model_content=tflite_model_bytes)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()
    out = interpreter.get_output_details()
    correct = 0
    for i in range(n_samples):
        data = np.expand_dims(x[i], axis=0).astype(np.float32)
        interpreter.set_tensor(inp[0]['index'], data)
        interpreter.invoke()
        pred = interpreter.get_tensor(out[0]['index'])
        if np.argmax(pred) == y[i]:
            correct += 1
        if (i + 1) % 100 == 0:
            print(f"    → {i+1}/{n_samples} images evaluated...")
    return correct / n_samples


def cosine_warmup_schedule(epoch, warmup_epochs, total_epochs, base_lr, target_lr):
    if epoch < warmup_epochs:
        return base_lr + (target_lr - base_lr) * (epoch / warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return target_lr * 0.5 * (1 + np.cos(np.pi * progress))


# ===========================================================
# SECTION 3: MODEL BUILDER
# ===========================================================

def build_efficientnet(trainable_base=False,
                       name="EfficientNetV2M",
                       augmentation_layer=None,
                       dropout_rate=0.3,
                       extra_dense=None):
    aug = augmentation_layer if augmentation_layer is not None else data_augmentation
    inp = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="flowers_input")
    x   = aug(inp)
    base = keras.applications.EfficientNetV2M(
        include_top = False,
        weights     = 'imagenet' if USE_PRETRAINED else None,
        pooling     = 'avg'
    )
    base.trainable = trainable_base
    out = base(x, training=False)
    out = layers.Dropout(dropout_rate, name="head_dropout")(out)
    if extra_dense:
        out = layers.Dense(extra_dense, activation='relu', name="extra_dense")(out)
        out = layers.Dropout(dropout_rate * 0.5, name="extra_dropout")(out)
    out = layers.Dense(NUM_CLASSES, activation='softmax', name="predictions")(out)
    model = keras.Model(inputs=inp, outputs=out, name=name)
    return model, base


# ===========================================================
# SECTION 4: TECHNIQUE 1 — BASELINE (2-Phase Training)
# ===========================================================

print("\n" + "=" * 70)
print("[3] BASELINE  —  EfficientNetV2M  (2-Phase Training)")
print("=" * 70)


def train_baseline():
    model, base = build_efficientnet(trainable_base=False, name="baseline_efficientnet")

    print(f"\n  Phase 1: Frozen base — training head ({PHASE1_EPOCHS} epochs, LR={PHASE1_LR})")
    model.compile(
        optimizer = keras.optimizers.Adam(PHASE1_LR),
        loss      = 'sparse_categorical_crossentropy',
        metrics   = ['accuracy']
    )
    model.fit(
        make_train_dataset(PHASE1_BATCH_SIZE),
        epochs          = PHASE1_EPOCHS,
        validation_data = make_test_dataset(PHASE1_BATCH_SIZE),
        callbacks       = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=3,
                restore_best_weights=True, verbose=0),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=2, verbose=1),
        ],
        verbose = 1
    )

    print(f"\n  Phase 2: Unfreezing last {FINE_TUNE_LAYERS} layers (LR={PHASE2_LR})")
    base.trainable = True
    freeze_until = max(0, len(base.layers) - FINE_TUNE_LAYERS)
    for layer in base.layers[:freeze_until]:
        layer.trainable = False

    model.compile(
        optimizer = keras.optimizers.Adam(PHASE2_LR),
        loss      = 'sparse_categorical_crossentropy',
        metrics   = ['accuracy']
    )
    history = model.fit(
        make_train_dataset(PHASE2_BATCH_SIZE),
        epochs          = PHASE2_EPOCHS,
        validation_data = make_test_dataset(PHASE2_BATCH_SIZE),
        callbacks       = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=ES_PATIENCE,
                restore_best_weights=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.3, patience=3, verbose=1),
        ],
        verbose = 1
    )

    _, acc = model.evaluate(make_test_dataset(PHASE2_BATCH_SIZE), verbose=0)
    model.save(BASELINE_MODEL_PATH)

    print(f"\n  ✅ Baseline Accuracy : {acc*100:.2f}%")
    print(f"  ✅ Model Size        : {get_model_size_mb(model):.2f} MB")

    return {
        'accuracy' : acc,
        'size'     : get_model_size_mb(model),
        'params'   : model.count_params(),
        'history'  : history
    }


print("\nStarting baseline training (this is the longest step)...")
baseline_result, baseline_co2, baseline_time = measure_energy(train_baseline, "Baseline")
baseline_result['co2']  = baseline_co2
baseline_result['time'] = baseline_time
save_result('Baseline', baseline_result)


# ===========================================================
# SECTION 5: TECHNIQUE 2 — QUANTIZATION (Float16 TFLite)
# ===========================================================

print("\n" + "=" * 70)
print("[4] QUANTIZATION  —  Float16 TFLite")
print("=" * 70)


def apply_quantization():
    print("\n  Loading baseline model...")
    model = keras.models.load_model(BASELINE_MODEL_PATH)

    print("  Converting to TFLite with Float16 quantization...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    with open(QUANTIZED_MODEL_PATH, 'wb') as f:
        f.write(tflite_model)

    q_size = len(tflite_model) / (1024 ** 2)
    print(f"  Quantized size    : {q_size:.3f} MB  (was {baseline_result['size']:.2f} MB)")
    print(f"  Compression ratio : {baseline_result['size']/q_size:.1f}x")

    print(f"\n  Evaluating on {QUANT_TEST_SAMPLES} samples...")
    q_acc = evaluate_tflite(tflite_model, x_test, y_test)

    print(f"\n  ✅ Quantized Accuracy : {q_acc*100:.2f}%")
    print(f"  ✅ Accuracy change    : {(q_acc - baseline_result['accuracy'])*100:+.2f}%")

    return {
        'accuracy' : q_acc,
        'size'     : q_size,
        'params'   : 'N/A'
    }


print("\nApplying quantization...")
quantized_result, quantized_co2, quantized_time = measure_energy(apply_quantization, "Quantization")
quantized_result['co2']  = quantized_co2
quantized_result['time'] = quantized_time
save_result('Quantization', quantized_result)


# ===========================================================
# SECTION 6: TECHNIQUE 3 — EARLY STOPPING
# ===========================================================

print("\n" + "=" * 70)
print("[5] EARLY STOPPING  —  EfficientNetV2M with auto-stop")
print("=" * 70)


def apply_early_stopping():
    model, base = build_efficientnet(trainable_base=False, name="early_stop_efficientnet")

    model.compile(
        optimizer = keras.optimizers.Adam(PHASE1_LR),
        loss      = 'sparse_categorical_crossentropy',
        metrics   = ['accuracy']
    )

    print(f"\n  Training with early stopping (max {PHASE1_EPOCHS + PHASE2_EPOCHS} epochs)...")
    history = model.fit(
        make_train_dataset(PHASE1_BATCH_SIZE),
        epochs          = PHASE1_EPOCHS + PHASE2_EPOCHS,
        validation_data = make_test_dataset(PHASE1_BATCH_SIZE),
        callbacks       = [
            keras.callbacks.EarlyStopping(
                monitor=ES_MONITOR, patience=ES_PATIENCE,
                restore_best_weights=ES_RESTORE_BEST, verbose=1),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=2, verbose=1),
        ],
        verbose = 1
    )

    _, acc = model.evaluate(make_test_dataset(PHASE1_BATCH_SIZE), verbose=0)
    model.save(EARLY_STOP_MODEL_PATH)

    epochs_used = len(history.history['loss'])
    print(f"\n  ✅ Early Stopping Accuracy : {acc*100:.2f}%")
    print(f"  ✅ Epochs used             : {epochs_used} / {PHASE1_EPOCHS + PHASE2_EPOCHS}")
    print(f"  ✅ Energy saved            : {PHASE1_EPOCHS + PHASE2_EPOCHS - epochs_used} epochs skipped")

    return {
        'accuracy'    : acc,
        'size'        : get_model_size_mb(model),
        'params'      : model.count_params(),
        'epochs_used' : epochs_used,
        'history'     : history
    }


print("\nApplying early stopping...")
early_result, early_co2, early_time = measure_energy(apply_early_stopping, "Early Stopping")
early_result['co2']  = early_co2
early_result['time'] = early_time
save_result('Early Stopping', early_result)


# ===========================================================
# SECTION 7: TECHNIQUE 4 — TRANSFER LEARNING (2-Phase)
# ===========================================================

print("\n" + "=" * 70)
print("[6] TRANSFER LEARNING  —  Frozen backbone + 2-phase training")
print("=" * 70)


def apply_transfer_learning():
    model, base = build_efficientnet(
        trainable_base    = False,
        name              = "transfer_learning",
        augmentation_layer= strong_augmentation,
        dropout_rate      = TL_DROPOUT,
        extra_dense       = TL_DENSE_UNITS
    )
    base.trainable = False

    print(f"\n  Phase 1: Frozen base — training head")
    print(f"  Head         : Dense({TL_DENSE_UNITS}) + Dropout + Softmax")
    print(f"  Augmentation : Strong")
    print(f"  Epochs       : {TL_EPOCHS}  |  LR: {TL_LR}  |  Batch: {TL_BATCH_SIZE}")

    model.compile(
        optimizer = keras.optimizers.Adam(TL_LR),
        loss      = 'sparse_categorical_crossentropy',
        metrics   = ['accuracy']
    )
    model.fit(
        make_train_dataset(TL_BATCH_SIZE),
        epochs          = TL_EPOCHS,
        validation_data = make_test_dataset(TL_BATCH_SIZE),
        callbacks       = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=5,
                restore_best_weights=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.4, patience=3,
                min_lr=1e-6, verbose=1),
        ],
        verbose = 1
    )

    PHASE2_UNFREEZE = 20
    PHASE2_BATCH    = 8

    print(f"\n  Phase 2: Unfreezing last {PHASE2_UNFREEZE} backbone layers (LR=1e-5)")
    print(f"  Batch size : {PHASE2_BATCH}")
    base.trainable = True
    freeze_until = max(0, len(base.layers) - PHASE2_UNFREEZE)
    for layer in base.layers[:freeze_until]:
        layer.trainable = False

    model.compile(
        optimizer = keras.optimizers.Adam(1e-5),
        loss      = 'sparse_categorical_crossentropy',
        metrics   = ['accuracy']
    )
    history = model.fit(
        make_train_dataset(PHASE2_BATCH),
        epochs          = 10,
        validation_data = make_test_dataset(PHASE2_BATCH),
        callbacks       = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=5,
                restore_best_weights=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.3, patience=3,
                min_lr=1e-7, verbose=1),
        ],
        verbose = 1
    )

    _, acc = model.evaluate(make_test_dataset(PHASE2_BATCH), verbose=0)
    model.save(TL_MODEL_PATH)

    print(f"\n  ✅ Transfer Learning Accuracy : {acc*100:.2f}%")
    print(f"  ✅ Model Size                 : {get_model_size_mb(model):.2f} MB")

    return {
        'accuracy' : acc,
        'size'     : get_model_size_mb(model),
        'params'   : model.count_params(),
        'history'  : history
    }


print("\nApplying transfer learning...")
tl_result, tl_co2, tl_time = measure_energy(apply_transfer_learning, "Transfer Learning")
tl_result['co2']  = tl_co2
tl_result['time'] = tl_time
save_result('Transfer Learning', tl_result)


# ===========================================================
# SECTION 8: TECHNIQUE 5 — FINE-TUNING (Deep Unfreeze)
# ===========================================================

print("\n" + "=" * 70)
print("[7] FINE-TUNING  —  Deep Layer Unfreeze + Cosine Warmup LR")
print("=" * 70)


def apply_finetuning():
    print("\n  Loading Transfer Learning model as starting point...")
    model = keras.models.load_model(TL_MODEL_PATH)

    base = None
    for layer in model.layers:
        if 'efficientnetv2' in layer.name.lower():
            base = layer
            break

    if base is None:
        print("  ⚠️  Backbone not found — unfreezing last model layers.")
        model.trainable = True
        freeze_until = max(0, len(model.layers) - FINETUNE_UNFREEZE_LAYERS)
        for layer in model.layers[:freeze_until]:
            layer.trainable = False
    else:
        base.trainable = True
        freeze_until = max(0, len(base.layers) - FINETUNE_UNFREEZE_LAYERS)
        for layer in base.layers[:freeze_until]:
            layer.trainable = False
        print(f"  Unfroze last {FINETUNE_UNFREEZE_LAYERS} backbone layers")

    print(f"  Trainable layers : {sum(1 for l in model.layers if l.trainable)}")
    print(f"  LR schedule      : Cosine warmup ({FINETUNE_WARMUP_EPOCHS} epochs)")
    print(f"  Total epochs     : {FINETUNE_EPOCHS}")

    def lr_schedule(epoch):
        return cosine_warmup_schedule(
            epoch,
            warmup_epochs = FINETUNE_WARMUP_EPOCHS,
            total_epochs  = FINETUNE_EPOCHS,
            base_lr       = FINETUNE_LR * 0.1,
            target_lr     = FINETUNE_LR
        )

    model.compile(
        optimizer = keras.optimizers.Adam(FINETUNE_LR),
        loss      = 'sparse_categorical_crossentropy',
        metrics   = ['accuracy']
    )

    print(f"\n  Fine-tuning ({FINETUNE_EPOCHS} epochs, base LR={FINETUNE_LR})...")
    history = model.fit(
        make_train_dataset(FINETUNE_BATCH_SIZE),
        epochs          = FINETUNE_EPOCHS,
        validation_data = make_test_dataset(FINETUNE_BATCH_SIZE),
        callbacks       = [
            keras.callbacks.LearningRateScheduler(lr_schedule, verbose=0),
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=ES_PATIENCE,
                restore_best_weights=True, verbose=1),
            keras.callbacks.ModelCheckpoint(
                FINETUNED_MODEL_PATH, monitor='val_accuracy',
                save_best_only=True, verbose=0),
        ],
        verbose = 1
    )

    if os.path.exists(FINETUNED_MODEL_PATH):
        model = keras.models.load_model(FINETUNED_MODEL_PATH)

    _, acc = model.evaluate(make_test_dataset(FINETUNE_BATCH_SIZE), verbose=0)

    print(f"\n  ✅ Fine-Tuned Accuracy  : {acc*100:.2f}%")
    print(f"  ✅ Model Size           : {get_model_size_mb(model):.2f} MB")
    print(f"  ✅ Improvement over TL  : {(acc - tl_result['accuracy'])*100:+.2f}%")

    return {
        'accuracy' : acc,
        'size'     : get_model_size_mb(model),
        'params'   : model.count_params(),
        'history'  : history
    }


print("\nApplying fine-tuning...")
finetuned_result, finetuned_co2, finetuned_time = measure_energy(apply_finetuning, "Fine-Tuning")
finetuned_result['co2']  = finetuned_co2
finetuned_result['time'] = finetuned_time
save_result('Fine-Tuning', finetuned_result)


# ===========================================================
# SECTION 9: RESULTS TABLE
# ===========================================================

all_results = {
    'Baseline'          : baseline_result,
    'Quantization'      : quantized_result,
    'Early Stopping'    : early_result,
    'Transfer Learning' : tl_result,
    'Fine-Tuning'       : finetuned_result,
}

print("\n" + "=" * 110)
print("  TECHNIQUE COMPARISON")
print("=" * 110)
print(f"{'Technique':<22} {'Accuracy':>10} {'Size (MB)':>12} "
      f"{'CO₂ (kg)':>14} {'Time (s)':>10} {'Params':>14}")
print("-" * 110)

for name, data in all_results.items():
    acc_s    = f"{data['accuracy']*100:.2f}%"  if data.get('accuracy', 0) > 0  else "N/A"
    size_s   = f"{data['size']:.3f}"           if data.get('size', 0) > 0      else "N/A"
    co2_s    = f"{data.get('co2',  0.0):.6f}"
    time_s   = f"{data.get('time', 0.0):.1f}"
    params_s = (f"{int(data['params']):,}"
                if data.get('params') and data['params'] not in ('N/A', 0) else 'N/A')
    print(f"{name:<22} {acc_s:>10} {size_s:>12} {co2_s:>14} {time_s:>10} {params_s:>14}")

print("=" * 110)

acc_entries = [(k, v) for k, v in all_results.items() if v.get('accuracy', 0) > 0]
if acc_entries:
    best = max(acc_entries, key=lambda x: x[1]['accuracy'])
    print(f"\n  🏆 Highest accuracy: {best[0]}  ({best[1]['accuracy']*100:.2f}%)")
else:
    print("\n  ⚠️  No techniques produced a valid accuracy.")

print("\n✅ train.py complete — passing results to analyze.py")
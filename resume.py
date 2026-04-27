# ===========================================================
# resume.py — Recovery script
# Loads saved metrics for all completed techniques:
#   Baseline, Quantization, Early Stopping, Transfer Learning
# Then runs ONLY Fine-Tuning.
# ===========================================================

print("=" * 70)
print("  RESUME — Loading saved results + running Fine-Tuning only")
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

for d in [RESULTS_DIR, MODELS_DIR, PLOTS_DIR, DATA_DIR]:
    os.makedirs(d, exist_ok=True)


# ===========================================================
# SECTION 1: DATA LOADING
# ===========================================================

print("\n[2] Loading dataset...")

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

print("  Warming dataset cache...")
TRAIN_SIZE = sum(1 for _ in ds_train_proc)
TEST_SIZE  = sum(1 for _ in ds_test_proc)
print(f"✅ Train: {TRAIN_SIZE}  |  Test: {TEST_SIZE}")

def make_train_dataset(batch_size, shuffle=True):
    ds = ds_train_proc
    if shuffle:
        ds = ds.shuffle(buffer_size=2048, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def make_test_dataset(batch_size):
    return ds_test_proc.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ===========================================================
# SECTION 2: HELPERS
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


def load_result(name):
    path = os.path.join(DATA_DIR, f"result_{name.replace(' ', '_')}.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        try:
            data['params'] = int(data['params'])
        except (ValueError, TypeError):
            pass
        return data
    return None


def load_or_warn(name):
    result = load_result(name)
    if result:
        print(f"  ✅ {name:<22} — accuracy: {result['accuracy']*100:.2f}%  "
              f"co2: {result['co2']:.6f} kg  time: {result['time']:.1f}s")
    else:
        print(f"  ⚠️  {name}: JSON not found — metrics will show 0.0")
        result = {'accuracy': 0.0, 'size': 0.0, 'params': 'N/A',
                  'co2': 0.0, 'time': 0.0}
    return result


def measure_energy(func, name):
    tracker = EmissionsTracker(
        output_dir=DATA_DIR, save_to_file=False,
        project_name=name, log_level='error'
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


def cosine_warmup_schedule(epoch, warmup_epochs, total_epochs, base_lr, target_lr):
    if epoch < warmup_epochs:
        return base_lr + (target_lr - base_lr) * (epoch / warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return target_lr * 0.5 * (1 + np.cos(np.pi * progress))


# ===========================================================
# SECTION 3: LOAD ALL COMPLETED RESULTS FROM JSON
# ===========================================================

print("\n" + "=" * 70)
print("  Loading all previously saved metrics from JSON files...")
print("=" * 70)

baseline_result  = load_or_warn('Baseline')
quantized_result = load_or_warn('Quantization')
early_result     = load_or_warn('Early Stopping')
tl_result        = load_or_warn('Transfer Learning')

print()


# ===========================================================
# SECTION 4: FINE-TUNING
# ===========================================================

print("\n" + "=" * 70)
print("[6] FINE-TUNING  —  Deep Layer Unfreeze + Cosine Warmup LR")
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
finetuned_result, finetuned_co2, finetuned_time = measure_energy(
    apply_finetuning, "Fine-Tuning")
finetuned_result['co2']  = finetuned_co2
finetuned_result['time'] = finetuned_time
save_result('Fine-Tuning', finetuned_result)


# ===========================================================
# SECTION 5: FULL COMBINED RESULTS TABLE
# ===========================================================

all_results = {
    'Baseline'          : baseline_result,
    'Quantization'      : quantized_result,
    'Early Stopping'    : early_result,
    'Transfer Learning' : tl_result,
    'Fine-Tuning'       : finetuned_result,
}

print("\n" + "=" * 110)
print("  FULL TECHNIQUE COMPARISON")
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
                if data.get('params') and data['params'] not in ('N/A', 0)
                else 'N/A')
    print(f"{name:<22} {acc_s:>10} {size_s:>12} {co2_s:>14} "
          f"{time_s:>10} {params_s:>14}")

print("=" * 110)

acc_entries = [(k, v) for k, v in all_results.items() if v.get('accuracy', 0) > 0]
if acc_entries:
    best = max(acc_entries, key=lambda x: x[1]['accuracy'])
    print(f"\n  🏆 Highest accuracy: {best[0]}  "
          f"({best[1]['accuracy']*100:.2f}%)")
else:
    print("\n  ⚠️  No techniques produced a valid accuracy.")

print("\n✅ resume.py complete — passing results to analyze.py")
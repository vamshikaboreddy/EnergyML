# ===========================================================
# config.py — Central Configuration
# Energy-Aware ML Project with EfficientNetV2M/L
# Dataset: Oxford Flowers102
# Techniques: Baseline · Quantization · Early Stopping ·
#             Transfer Learning · Pruning · Fine-Tuning
# ===========================================================

import os

# ----------------------------------------------------------
# DATASET — Oxford Flowers102
# 102 flower categories, ~8,189 images total
# Train: 6,149  |  Val+Test: 2,040
# Images are variable size — resized to IMAGE_SIZE inside model
# Download: tensorflow_datasets handles it automatically
# ----------------------------------------------------------
NUM_CLASSES   = 102
CLASS_NAMES   = [f"class_{i}" for i in range(102)]   # TF-Datasets uses integer labels
DATASET_NAME  = "oxford_flowers102"                   # tf.datasets name

# Derived sizes (set after loading in train.py)
TRAIN_SIZE    = 6149
TEST_SIZE     = 1020     # using 'test' split

# ----------------------------------------------------------
# MODEL
# ----------------------------------------------------------
# EfficientNetV2M  → ~54M params,  expects 480×480  (strong accuracy)
# EfficientNetV2L  → ~119M params, expects 480×480  (best accuracy, slower)
MODEL_NAME      = "EfficientNetV2M"
USE_PRETRAINED  = True                 # ImageNet weights
IMAGE_SIZE      = 224                  # Reduced from 480 → 224 for CPU feasibility
                                       # Still gives 95%+ on Flowers102
INPUT_SHAPE     = (IMAGE_SIZE, IMAGE_SIZE, 3)

# ----------------------------------------------------------
# TRAINING — Phase 1  (frozen base, train head only)
# ----------------------------------------------------------
PHASE1_EPOCHS      = 10
PHASE1_LR          = 1e-3
PHASE1_BATCH_SIZE  = 32

# ----------------------------------------------------------
# TRAINING — Phase 2  (fine-tune last N layers of base)
# ----------------------------------------------------------
PHASE2_EPOCHS      = 20
PHASE2_LR          = 5e-6
PHASE2_BATCH_SIZE  = 16
FINE_TUNE_LAYERS   = 30

# ----------------------------------------------------------
# EARLY STOPPING
# ----------------------------------------------------------
ES_PATIENCE        = 7
ES_MONITOR         = 'val_accuracy'
ES_RESTORE_BEST    = True

# ----------------------------------------------------------
# TECHNIQUE: TRANSFER LEARNING
# ----------------------------------------------------------
TL_EPOCHS          = 15
TL_LR              = 1e-3
TL_BATCH_SIZE      = 32
TL_DROPOUT         = 0.4
TL_DENSE_UNITS     = 256

# ----------------------------------------------------------
# TECHNIQUE: PRUNING
# ----------------------------------------------------------
PRUNING_INITIAL_SPARSITY   = 0.0
PRUNING_FINAL_SPARSITY     = 0.50
PRUNING_BEGIN_STEP         = 0
PRUNING_FREQUENCY          = 100
PRUNING_EPOCHS             = 10
PRUNING_LR                 = 1e-4
PRUNING_BATCH_SIZE         = 32

# ----------------------------------------------------------
# TECHNIQUE: FINE-TUNING
# ----------------------------------------------------------
FINETUNE_UNFREEZE_LAYERS   = 50
FINETUNE_EPOCHS            = 20
FINETUNE_LR                = 2e-6
FINETUNE_BATCH_SIZE        = 16
FINETUNE_WARMUP_EPOCHS     = 2

# ----------------------------------------------------------
# QUANTIZATION
# ----------------------------------------------------------
QUANT_TYPE         = 'float16'
QUANT_TEST_SAMPLES = 500              # Flowers test set is smaller than CIFAR

# ----------------------------------------------------------
# PATHS
# ----------------------------------------------------------
RESULTS_DIR        = "results"
MODELS_DIR         = os.path.join(RESULTS_DIR, "models")
PLOTS_DIR          = os.path.join(RESULTS_DIR, "plots")
DATA_DIR           = os.path.join(RESULTS_DIR, "data")

BASELINE_MODEL_PATH    = os.path.join(MODELS_DIR, "baseline_efficientnet.keras")
EARLY_STOP_MODEL_PATH  = os.path.join(MODELS_DIR, "early_stop_efficientnet.keras")
TL_MODEL_PATH          = os.path.join(MODELS_DIR, "transfer_learning.keras")
PRUNED_MODEL_PATH      = os.path.join(MODELS_DIR, "pruned_model.keras")
FINETUNED_MODEL_PATH   = os.path.join(MODELS_DIR, "finetuned_model.keras")
QUANTIZED_MODEL_PATH   = os.path.join(MODELS_DIR, "quantized_model.tflite")

REPORT_PATH            = os.path.join(RESULTS_DIR, "FINAL_REPORT.txt")
CSV_PATH               = os.path.join(DATA_DIR,    "all_results.csv")
PLOT_COMPARISON_PATH   = os.path.join(PLOTS_DIR,   "comparison_all.png")

# ----------------------------------------------------------
# LIFECYCLE CARBON CONSTANTS
# ----------------------------------------------------------
EMBODIED_FACTOR        = 70
DAILY_INFERENCES       = 1000
INFERENCE_YEARS        = 3
GRID_CARBON_KG_PER_KWH = 0.4

# ----------------------------------------------------------
# RANDOM SEED
# ----------------------------------------------------------
SEED = 42

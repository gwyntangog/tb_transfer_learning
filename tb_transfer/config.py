"""Fixed protocol values for the CheXpert-to-TBX11K runs."""

from __future__ import annotations

from pathlib import Path

PROTOCOL = "chexpert-tb-v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "outputs"
DEFAULT_CHEXPERT_ROOT = Path(r"C:\data\chexpert\CheXpert-v1.0")

ARCHS = ("resnet50", "densenet121", "vit_b_16")
SPLIT_SEED = 0
STAGE_A_SEEDS = (0, 1, 2)
STAGE_B_SEED = 0
STAGE_A_LR = 1e-4
STAGE_B_LR = 1e-5
WEIGHT_DECAY = 0.01
MAX_EPOCHS = 20
STAGE_A_PATIENCE = 3
STAGE_B_PATIENCE = 1
BATCH_SIZE = 32
IMAGE_SIZE = 224

TBX_TRAIN_COUNTS = {"tb": 600, "healthy": 3000, "sick": 3000}
TBX_VAL_COUNTS = {"tb": 200, "healthy": 800, "sick": 800}
TBX_HOLDOUT_COUNTS = {"tb": 60, "healthy": 300, "sick": 300}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

STAGE_A_FIELDS = (
    "architecture",
    "init",
    "seed",
    "epoch",
    "epochs_ran",
    "selection_auprc",
    "batch_size",
    "lr",
    "n_fit",
    "n_sel",
    "n_pos_fit",
    "n_pos_sel",
    "n_patients_fit",
    "n_patients_sel",
    "selection_auroc",
    "selection_threshold",
    "selection_sensitivity",
    "selection_specificity",
    "selection_youden_j",
    "selection_tp",
    "selection_fp",
    "selection_tn",
    "selection_fn",
    "device",
    "selected",
)
STAGE_B_FIELDS = (
    "architecture",
    "init",
    "seed",
    "epoch",
    "epochs_ran",
    "selection_auprc",
    "threshold",
    "val_auprc",
    "val_auprc_no_sick",
    "val_auroc",
    "val_auroc_no_sick",
    "fpr_healthy",
    "fpr_sick",
    "val_sensitivity",
    "val_specificity",
    "val_specificity_healthy",
    "val_specificity_sick",
    "val_ppv",
    "val_npv",
    "val_youden_j",
    "val_tp",
    "val_fp",
    "val_tn",
    "val_fn",
    "val_fp_healthy",
    "val_tn_healthy",
    "val_fp_sick",
    "val_tn_sick",
    "n_val_tb",
    "n_val_healthy",
    "n_val_sick",
    "holdout_auprc",
    "holdout_auroc",
    "holdout_sensitivity",
    "holdout_specificity",
    "holdout_youden_j",
    "holdout_tp",
    "holdout_fp",
    "holdout_tn",
    "holdout_fn",
    "holdout_fpr_healthy",
    "holdout_fpr_sick",
    "n_holdout_tb",
    "n_holdout_healthy",
    "n_holdout_sick",
    "batch_size",
    "lr",
    "n_fit",
    "n_holdout",
    "n_val",
    "device",
)

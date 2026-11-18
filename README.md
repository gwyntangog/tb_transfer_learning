# TB transfer learning

Stage A trains ResNet-50, DenseNet-121, and ViT-B/16 on Stanford CheXpert frontal films with an explicit pneumonia label (`Pneumonia` 1.0 or 0.0). Stage B fine-tunes those checkpoints, and a fresh ImageNet initialization, on the official TBX11K train list.

The tests do not need either dataset:

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests
```

## Data

CheXpert for this machine lives at `C:\data\chexpert\CheXpert-v1.0`. That folder contains `train.csv`. Extracted films are `train\patient...\view.jpg`, while `train.csv` stores paths that start with `CheXpert-v1.0/train/...`. The loader checks both.

TBX11K is separate. Set `TBX11K_ROOT` to the folder that contains `lists\` and `imgs\`. The code reads only `lists\TBX11K_train.txt` and `lists\TBX11K_val.txt`.

Both roots are outside this repo. `.gitignore` is the backstop if a copy is placed in the workspace: images, `train.csv`, checkpoints, `outputs/`, `all_*.txt`, and `fetch_frontal_pneumonia.py`. That download script holds an Azure SAS URL, so leave it in `C:\data\chexpert` and do not copy it here.

```powershell
$env:CHEXPERT_ROOT = "C:\data\chexpert\CheXpert-v1.0"
$env:TBX11K_ROOT = "D:\datasets\TBX11K"
```

`CHEXPERT_ROOT` already defaults to `C:\data\chexpert\CheXpert-v1.0`.

## Check, then train

`--check` resolves files and writes the frozen splits. It does not train, and Stage B's check does not open the val list.

```powershell
python -m tb_transfer.train_a --check
python -m tb_transfer.train_b --check
python -m tb_transfer.train_a
python -m tb_transfer.train_b
```

Stage A uses AdamW at `1e-4`, seeds 0, 1, and 2, patience 3, and a cap of 20 epochs. The patient split is seed 0 for every seed. The best seed per architecture is `outputs\stage_a\<arch>\best.pt`.

Stage B uses AdamW at `1e-5` on every layer, a new one-logit head, patience 1, and the same 20-epoch cap. The stratified 10% holdout is written to `outputs\stage_b\tbx11k_holdout_seed0.json` and reused. The Youden threshold comes from that holdout. `TBX11K_val.txt` is scored once per run after that.

A finished seed or fine-tune is reused. Delete its folder under `outputs`, or pass `--force`, to retrain it. If CUDA runs out of memory at batch 32, that run restarts at batch 16 and the table records the batch size.

## Report

`outputs\stage_b_metrics.csv` has, for each architecture and each initialization:

- selection AUPRC, chosen epoch, and Youden threshold
- val AUPRC on all 1,800 films
- val AUPRC after the 800 sick films are removed
- false-positive rate on the healthy films and on the sick films

Copy those numbers into the writeup. The csv is under `outputs/`, which is gitignored.

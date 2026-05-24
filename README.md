# Dog Breed Classification Assignment

PyTorch Lightning project for dog breed image classification using the Kaggle dataset:

https://www.kaggle.com/datasets/khushikhushikhushi/dog-breed-image-dataset

The Docker image installs this package with `pip install -e .`, so the scripts and package imports work inside the container.

## Project Layout

```text
assignment/
  Dockerfile
  .devcontainer/devcontainer.json
  pyproject.toml
  train.py
  eval.py
  infer.py
  src/dog_breed_assignment/
  data/
  logs/
  samples/
  predictions/
```

Runtime folders are intentionally used with volume mounts:

```text
data/         Kaggle dataset files
logs/         Lightning logs and checkpoints
samples/      input images for inference
predictions/  inference output images
```

## Kaggle Credentials

Use either environment variables:

```bash
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_key"
```

Or mount a Kaggle token file:

```text
~/.kaggle/kaggle.json
```

## Build Docker Image

Run from this `assignment/` folder:

```bash
docker build -t dog-breed-assignment .
```

## Train With Docker

This command mounts data and logs so downloaded data and checkpoints stay on the host:

```bash
docker run --rm \
  -e KAGGLE_USERNAME="$KAGGLE_USERNAME" \
  -e KAGGLE_KEY="$KAGGLE_KEY" \
  -v "$PWD/data:/workspace/assignment/data" \
  -v "$PWD/logs:/workspace/assignment/logs" \
  dog-breed-assignment \
  python train.py
```

For a small CPU smoke run:

```bash
docker run --rm \
  -e KAGGLE_USERNAME="$KAGGLE_USERNAME" \
  -e KAGGLE_KEY="$KAGGLE_KEY" \
  -v "$PWD/data:/workspace/assignment/data" \
  -v "$PWD/logs:/workspace/assignment/logs" \
  dog-breed-assignment \
  python train.py --limit_train_batches 20 --limit_val_batches 10
```

## Eval With Docker

Replace the checkpoint path with one from `logs/dog_breed_classification/.../checkpoints/`.

```bash
docker run --rm \
  -e KAGGLE_USERNAME="$KAGGLE_USERNAME" \
  -e KAGGLE_KEY="$KAGGLE_KEY" \
  -v "$PWD/data:/workspace/assignment/data" \
  -v "$PWD/logs:/workspace/assignment/logs" \
  dog-breed-assignment \
  python eval.py \
    --ckpt_path logs/dog_breed_classification/version_0/checkpoints/last.ckpt
```

`eval.py` prints validation metrics, including `val_loss`, `val_acc`, `correct`, and `total`.

## Infer On 10 Images With Docker

Put at least 10 images in `samples/`, then run:

```bash
docker run --rm \
  -v "$PWD/samples:/workspace/assignment/samples" \
  -v "$PWD/predictions:/workspace/assignment/predictions" \
  -v "$PWD/logs:/workspace/assignment/logs" \
  dog-breed-assignment \
  python infer.py \
    --input_folder samples \
    --output_folder predictions \
    --ckpt_path logs/dog_breed_classification/version_0/checkpoints/last.ckpt \
    --limit 10
```

Prediction images are written to `predictions/` as `*_prediction.png`.

## Local Commands Without Docker

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
python train.py
python eval.py --ckpt_path logs/dog_breed_classification/version_0/checkpoints/last.ckpt
python infer.py --input_folder samples --output_folder predictions --ckpt_path logs/dog_breed_classification/version_0/checkpoints/last.ckpt --limit 10
```

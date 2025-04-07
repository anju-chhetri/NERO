# Default values
DATASET="gastrovision"
BATCH_SIZE=32
EPOCHS=20
VAL_ITER=1
NUM_CLASSES=11
LR=0.0001
MODEL_NAME="resnet18"
TORCH_PATH="/scratch/achhetri/aischool"
CHECKPOINT_DIR="/scratch/achhetri/experimentalResults/g-ood/"
TRAIN_DIR="/work/FAC/HEC/DESI/yshresth/aim/achhetri/medical/$DATASET/ID/train"
TEST_DIR="/work/FAC/HEC/DESI/yshresth/aim/achhetri/medical/$DATASET/ID/test"
SEED=42
python3 model_training/train.py --model_name "$MODEL_NAME" --batch_size "$BATCH_SIZE" --epochs "$EPOCHS" --dataset "$DATASET" --num_classes "$NUM_CLASSES" --torch_path "$TORCH_PATH" --checkpoint_dir "$CHECKPOINT_DIR" --train_dir "$TRAIN_DIR" --test_dir "$TEST_DIR" --seed "$SEED"
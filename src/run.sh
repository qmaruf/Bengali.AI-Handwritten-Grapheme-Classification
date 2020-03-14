# export PYTHONWARNINGS="ignore"
export CUDA_VISIBLE_DEVICES=0,1
export IMG_HEIGHT=64
export IMG_WIDTH=64
export EPOCHS=32
export LEARNING_RATE=0.0001
export TRAIN_BATCH_SIZE=128
export VALID_BATCH_SIZE=64
export MODEL_MEAN="(0.485,0.456,0.406)"
export MODEL_STD="(0.229,0.224,0.225)"
export BASE_MODEL='resnet50'
export TRAINING_FOLDS_CSV='../input/train_folds.csv'

export TRAINING_FOLDS="(0,1,2,3)"
export VALIDATION_FOLDS="(4,)"
python train.py

export TRAINING_FOLDS="(0,1,2,4)"
export VALIDATION_FOLDS="(3,)"
python train.py

export TRAINING_FOLDS="(0,1,3,4)"
export VALIDATION_FOLDS="(2,)"
python train.py

export TRAINING_FOLDS="(0,2,3,4)"
export VALIDATION_FOLDS="(1,)"
python train.py

export TRAINING_FOLDS="(1,2,3,4)"
export VALIDATION_FOLDS="(0,)"
python train.py
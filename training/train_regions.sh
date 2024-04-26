export nnUNet_results="nnUNet_training/nnUNet_results"
export nnUNet_raw="nnUNet_training/nnUNet_raw"
export nnUNet_preprocessed="nnUNet_training/nnUNet_preprocessed"
export nnUNet_predictions="nnUNet_training/nnUNet_predictions"

GPU_ID=0 # Put the GPU IDs here
DATASET=557
TRAINER=nnUNetTrainer

nnUNetv2_plan_and_preprocess -d $DATASET -c 2d -np 8 --verify_dataset_integrity

for fold in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=$GPU_ID nnUNetv2_train $DATASET 2d $fold -tr $TRAINER --npz -num_gpus 1
done
nnUNetv2_find_best_configuration $DATASET -c 2d -tr $TRAINER

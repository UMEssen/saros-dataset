export nnUNet_results="nnUNet_training/nnUNet_results"
export nnUNet_raw="nnUNet_training/nnUNet_raw"
export nnUNet_preprocessed="nnUNet_training/nnUNet_preprocessed"
export nnUNet_predictions="nnUNet_training/nnUNet_predictions"
export nnUNet_eval="nnUNet_training/nnUNet_eval"

GPU_ID=0 # Put the GPU IDs here
DATASET=557
DATASET_NAME=Dataset557_BCA_2d_regions
TRAINER=nnUNetTrainer

CUDA_VISIBLE_DEVICES=$GPU_ID nnUNetv2_predict -d $DATASET -i $nnUNet_eval/$DATASET_NAME/imagesTs -o $nnUNet_predictions/$DATASET_NAME/test -tr $TRAINER -c 2d -p nnUNetPlans

python training/evaluate.py --gt-folder $nnUNet_eval/$DATASET_NAME/labelsTs --pred-folder $nnUNet_predictions/${DATASET_NAME}/test --dataset regions --results-folder results --ignore-label 255

nnUNetv2_apply_postprocessing -i $nnUNet_predictions/$DATASET_NAME/test -o $nnUNet_predictions/$DATASET_NAME/test_pp -pp_pkl_file $nnUNet_results/$DATASET_NAME/${TRAINER}__nnUNetPlans__2d/crossval_results_folds_0_1_2_3_4/postprocessing.pkl -np 8 -plans_json $nnUNet_results/$DATASET_NAME/${TRAINER}__nnUNetPlans__2d/crossval_results_folds_0_1_2_3_4/plans.json

python training/evaluate.py --gt-folder $nnUNet_eval/$DATASET_NAME/labelsTs --pred-folder $nnUNet_predictions/${DATASET_NAME}/test_pp --dataset regions --results-folder results --ignore-label 255

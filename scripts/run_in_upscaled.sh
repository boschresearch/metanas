


DATASET=miniimagenet
DATASET_DIR=/path/to/miniimagenet
TRAIN_DIR=/results/path
		

mkdir -p $TRAIN_DIR

args=(
    # Execution
    --name run_in \
    --job_id 0 \
    --path ${TRAIN_DIR} \
    --data_path ${DATASET_DIR} \
    --dataset $DATASET
    --seed 1 \
    --hp_setting 'in' \
    --use_hp_setting 1 \
    --workers 0 \
    --gpus 0 \


    # few shot params
    # examples per class
    --n 5 \
    # number classes  
    --k 5 \
    # test examples per class
    --q 1 \

    # Meta Learning
    --meta_model metanas_in_v2 \
    --meta_epochs 100000 \
    --eval_freq 5000 \
    --eval_epochs 200 \
    --test_task_train_steps 50 \
    --w_weight_decay 0.0001  \
    --drop_path_prob 0.2 \
    --init_channels 96 \
    --use_torchmeta_loader \

)

python -u -m ../metanas.metanas_main "${args[@]}"



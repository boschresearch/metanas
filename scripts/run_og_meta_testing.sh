

DATASET=omniglot
DATASET_DIR=/path/to/omniglot
TRAIN_DIR=/results/path
		
mkdir -p $TRAIN_DIR

MODEL_PATH=/path/to/checkpoint/from/metatrain




args=(
    # Execution
    --name metatest_og \
    --job_id 0 \
    --path ${TRAIN_DIR} \
    --data_path ${DATASET_DIR} \
    --dataset $DATASET
    --hp_setting 'og_metanas' \
    --use_hp_setting 1 \
    --workers 0 \
    --gpus 0 \
    --test_adapt_steps 0.5 \
    --test_task_train_steps 100 \
    --eval \
    # few shot params
     # examples per class
    --n 5 \
    # number classes  
    --k 20 \
    # test examples per class
    --q 1 \

    --meta_model_prune_threshold 0.01 \
    --alpha_prune_threshold 0.05 \
    # Meta Learning
    --meta_model searchcnn \
    --model_path ${MODEL_PATH}
    --meta_epochs 30000 \
    --warm_up_epochs 15000 \
    --use_pairwise_input_alphas \
    --eval_freq 2500 \
    --eval_epochs 200 \


    --normalizer softmax \
    --normalizer_temp_anneal_mode linear \
    --normalizer_t_min 0.1 \
    --normalizer_t_max 1.0 \
    --drop_path_prob 0.2 \

    # Architectures
    --init_channels 28 \
    --layers 4 \
    --reduction_layers 1 3 \
    --use_first_order_darts \

    --use_torchmeta_loader \

)


python -u -m ../metanas.metanas_main "${args[@]}"


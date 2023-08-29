CUDA_VISIBLE_DEVICES=1 \
python train_biencoder_with_mlm_origin_code.py \
    --train_data_path data/tri_train/train_v8_small.json \
    --valid_data_path data/tri_train/valid_v8_small.json \
    --output exp.distbert.b16.tri.debug/ \
    --log_dir log.diff/ \
    --do_train \
    --do_eval \
    --learning_rate 2e-5 \
    --fp16 \
    --evaluate_during_training \
    --num_train_epochs 100 \
    --save_steps 0.5 \
    --logging_steps 0.01 \
    --save_total_limit 5 \
    --model_type distilbert \
    --model_name_or_path distilbert-base-uncased \
    --batch_size 16 \
    --mlm \
    --star_list_file data/star.list.220802.1-2 \

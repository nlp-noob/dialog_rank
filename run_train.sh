CUDA_VISIBLE_DEVICES=2 \
python train_biencoder_with_mlm.py \
    --order_file data/orders.star1-2.1.json \
    --output exp.distbert.b16 \
    --do_eval \
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
    --other_cluster_num 0 \

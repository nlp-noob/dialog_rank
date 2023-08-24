# sh train_msc_bert.sh log/disbert.stars.3 data/orders.stars.3.json distilbert distilbert-base-uncased
CUDA_VISIBLE_DEVICES=2 python train_biencoder_with_mlm.py -i data/orders.star1-2.1.json --output exp.distbert.b16 --do_train --do_eval --evaluate_during_training --num_train_epochs 100 --save_steps 0.5 --logging_steps 0.01 --save_total_limit 5 --model_type distilbert --model_name_or_path distilbert-base-uncased --batch_size 16 --mlm --star_list_file data/star.list.220802.1-2  
CUDA_VISIBLE_DEVICES=2 python train_biencoder_with_mlm.py -i data/orders.star1-2.1.json --output exp.distbert.b16.wcat.amp --do_train --do_eval --evaluate_during_training --num_train_epochs 30 --save_steps 0.5 --logging_steps 0.01 --save_total_limit 5 --model_type distilbert --model_name_or_path distilbert-base-uncased --batch_size 16 --mlm --star_list_file data/star.list.220802.1-2 --fp16
CUDA_VISIBLE_DEVICES=2 python train_biencoder_with_mlm.py -i data/orders.star1-2.json --output exp.1-2.bert.b8.wcat.amp --do_train --do_eval --evaluate_during_training --num_train_epochs 30 --save_steps 0.5 --logging_steps 0.01 --save_total_limit 5 --model_type bert --model_name_or_path bert-base-uncased --batch_size 8 --mlm --star_list_file data/star.list.220802.1-2 --fp16
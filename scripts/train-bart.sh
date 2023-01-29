nohup python trainer.py \
--max_seq_len 512 \
--max_squad_size 80000 \
--num_train_epochs 4 \
--eval_steps 1000 \
--lr_scheduler_type cosine \
--learning_rate 5e-5 \
--warmup_ratio 0.45 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--save_total_limit 1 \
--model_base  facebook/bart-base \
--run_id bart_base_model_1 \
--output_dir trained_models/  >> training_logs_bart5.out &
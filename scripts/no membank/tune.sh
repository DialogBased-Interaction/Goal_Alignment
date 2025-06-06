deepspeed fastchat/train/train_lora.py 
--model_name_or_path lmsys/vicuna-13b-v1.5
--lora_r 32 
--lora_alpha 64 
--lora_dropout 0.05 
--data_path ./data/train.json 
--eval_data_path ./data/eval.json 
--output_dir ./checkpoints13bnewdata 
--do_train True
--do_eval True
--num_train_epochs 4 
--fp16 True 
--per_device_train_batch_size 8 
--per_device_eval_batch_size 8 
--gradient_accumulation_steps 2 
--evaluation_strategy epoch 
--save_strategy epoch 
--learning_rate 2e-4 
--weight_decay 0. 
--warmup_ratio 0.03 
--lr_scheduler_type "cosine" 
--logging_strategy "steps" 
--logging_steps 1 
--tf32 True 
--model_max_length 2048 
--q_lora False 
--deepspeed playground/deepspeed_config_s2.json 
--gradient_checkpointing True 
--flash_attn False 
--report_to wandb 
--run_name vicuna-13bnewdata
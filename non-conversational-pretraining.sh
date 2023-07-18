source_dataset_name=$1
source_dataset_config=$2
max_history_turns=$3

prompt_config=$4
question_mask_length=$5

model_name_or_path=$6
checkpoint_dir=$7

command_line="python main.py --dataset_name $source_dataset_name --dataset_config $source_dataset_config --max_history_turns $max_history_turns \
--prompt_config $prompt_config --question_mask_length $question_mask_length \
--model_name_or_path $model_name_or_path --output_dir $checkpoint_dir \
--do_train --num_train_epochs 4  --optim adamw_torch --learning_rate 1e-4 \
--per_device_train_batch_size 8 --gradient_accumulation_steps 4 --per_device_eval_batch_size 16 \
--max_source_length 768 --max_target_length 25 --generation_max_length 25 \
--overwrite_output_dir --overwrite_cache \
--do_eval --predict_with_generate --load_best_model_at_end --metric_for_best_model bleu \
--save_strategy steps --save_steps 1000 --evaluation_strategy steps --eval_steps 1000 --eval_delay 5000 --save_total_limit 5"

echo -e "\n$command_line"
$command_line

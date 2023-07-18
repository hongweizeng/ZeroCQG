target_dataset_name=$1
max_history_turns=$2

prompt_config=$3
question_mask_length=$4

resume_from_checkpoint=$5
output_dir=$6

command_line="python main.py --dataset_name $target_dataset_name --max_history_turns $max_history_turns \
--prompt_config $prompt_config --question_mask_length $question_mask_length \
--model_name_or_path $resume_from_checkpoint --output_dir $output_dir \
--do_predict --per_device_eval_batch_size 8 \
--predict_with_generate --max_source_length 768 --max_target_length 30 \
--generation_max_length 15 \
--num_beams 5 --logging_steps 100 --overwrite_output_dir --overwrite_cache"

echo -e "\n$command_line"
$command_line

nlg-eval --hypothesis $output_dir/hypothesis.txt --references $output_dir/reference.txt
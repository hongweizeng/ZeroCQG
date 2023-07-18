
source_dataset_name="SQuAD"

model_name_or_path="t5-base"
prompt_config="FP"


for max_source_history_turn in 15
do

  for question_mask_length in 40
  do
    for source_dataset_config in "CKT" # "CKT+LD" #    "CKT+LD" #  "FP+DiffQM" "FP-SP" "FP-QM" "FP-TQM" "FP-SQM"
    do
      # 1. PRE-TRAINING on Non-conversational Dataset.

      source_path_suffix="$source_dataset_name-$source_dataset_config-$max_source_history_turn-$prompt_config-$question_mask_length-$model_name_or_path"

      checkpoint_dir="checkpoints/$source_path_suffix"

      echo -e "\n1. *** NON-CONVERSATIONAL PRE-TRAINING with $source_path_suffix"

      bash non-conversational-pretraining.sh $source_dataset_name $source_dataset_config $max_source_history_turn $prompt_config $question_mask_length $model_name_or_path $checkpoint_dir

      # 2. ZERO-SHOT INFERENCE on Conversational Dataset.
      for target_dataset_name in "CoQA" #  "CoQA" "QuAC" "QuAC"  # "CoQA"  # "DoQA" #  #
      do
        if [[ $target_dataset_name == "CoQA" ]]; then
          max_target_history_turn=9
        elif [[ $target_dataset_name == "QuAC" ]]; then
          max_target_history_turn=5
        else
          max_target_history_turn=3
        fi

        resume_from=$checkpoint_dir

        target_path_suffix="$target_dataset_name-$max_target_history_turn-$prompt_config-$question_mask_length"

        output_dir="outputs/$source_path_suffix/$target_path_suffix"

        echo -e "\n2. *** ZERO-SHOT CONVERSATIONAL INFERENCE on $target_path_suffix from $resume_from"

        bash conversational-inference.sh $target_dataset_name $max_target_history_turn $prompt_config $question_mask_length $resume_from $output_dir

      done
    done
  done
done
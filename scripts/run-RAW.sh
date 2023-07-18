
DEFAULT_HISTORY_SELECTION=NULL
DEFAULT_MAX_HISTORY_TURNS=0
DEFAULT_PROMPT_OPTION="NoP"

for SOURCE_DATASET_NAME in "SQuAD"
do
  for model_name_or_path in "t5-base" "facebook/bart-base" "google/pegasus-x-base"
  do
    # 1. PRE-TRAINING on Non-conversational Dataset.
    source_dataset_config="RAW"

    checkpoint_dir="checkpoints/$SOURCE_DATASET_NAME-$source_dataset_config-$model_name_or_path"

    echo -e "\n *** NON-CONVERSATIONAL PRE-TRAINING on $SOURCE_DATASET_NAME-$source_dataset_config with $model_name_or_path"

    bash scripts/non-conversational-pretraining.sh $SOURCE_DATASET_NAME $source_dataset_config $model_name_or_path $checkpoint_dir

    # 2. ZERO-SHOT INFERENCE on Conversational Dataset.
    for TARGET_DATASET_NAME in "CoQA" "QuAC"
    do
#      target_dataset_config="$DEFAULT_MAX_HISTORY_TURNS-NoP"
      target_dataset_config="0-NoP"

      resume_from=$checkpoint_dir

      output_dir="outputs/$source_dataset_name-$source_dataset_config-$model_name_or_path/$TARGET_DATASET_NAME-$target_dataset_config"

      echo -e "\n ZERO-SHOT CONVERSATIONAL INFERENCE on $TARGET_DATASET_NAME-$target_dataset_config from $resume_from output_dir=$output_dir"

      bash scripts/zero-shot-conversational-inference.sh $TARGET_DATASET_NAME $target_dataset_config $resume_from $output_dir

    done
  done
done
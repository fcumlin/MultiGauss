#!/bin/bash


# Pass the base directory as the first argument, for the NISQA wav files, it
# should be `path/to/dir/NISQA_Corpus`.
base_dir="$1"

if [ -z "$base_dir" ]; then
    echo "Error: No base directory provided."
    echo "Usage: $0 /path/to/NISQA_Corpus"
    exit 1
fi

for dataset in \
    "NISQA_TRAIN_SIM" \
    "NISQA_VAL_SIM" \
    "NISQA_TRAIN_LIVE" \
    "NISQA_VAL_LIVE" \
    "NISQA_TEST_LIVETALK" \
    "NISQA_TEST_P501" \
    "NISQA_TEST_FOR"
do
    python "generate_ssl_features.py" \
        --model_name "w2v2_xlsr_2b" \
        --data_dir "$base_dir/$dataset/deg" \
        --target_duration 10 \
        --layers 11 
    
    if [ $? -ne 0 ]; then
        echo "Error processing dataset: $dataset"
        exit 1
    fi
    echo "Successfully processed dataset: $dataset"
    echo "Waiting for 1 seconds before processing the next dataset..."
    sleep 1
    echo "Continuing to the next dataset..."
done

#!/bin/bash

# ==============================================================================
# Batch Download Script for Hugging Face Models
#
# Description:
#   - Iterates through a specified list of Hugging Face Model IDs.
#   - Downloads each model to a local directory using the huggingface-cli tool.
#
# Usage:
#   1. Edit the 'MODELS_TO_DOWNLOAD' array in the "Configuration" section below.
#   2. (Optional) Set the target download directory (TARGET_DIR).
#   3. (Optional) If you need to download private or gated models, provide your
#      Hugging Face Token (HF_TOKEN). A token with "read" permissions is recommended.
#   4. Save the file, then make it executable in your terminal: chmod +x download_models.sh
#   5. Run the script: ./download_models.sh
# ==============================================================================

# --- Configuration ---

# Add the Model IDs you want to download to this array.
# Example:
# MODELS_TO_DOWNLOAD=(
#   "bert-base-uncased"
#   "google/gemma-7b-it"
#   "meta-llama/Llama-2-7b-chat-hf"
#   "THUDM/chatglm3-6b"
# )
MODELS_TO_DOWNLOAD=(
  "Qwen/Qwen2.5-7B"
  "Qwen/Qwen2.5-7B-Instruct"
  "Qwen/Qwen2.5-Math-7B"
  "Qwen/Qwen2.5-Coder-7B-Instruct"
  "WangCa/Qwen2.5-7B-Medicine"
  "huihui-ai/Qwen2.5-7B-Instruct-abliterated-v2"
  "Locutusque/StockQwen-2.5-7B"
  "bunnycore/QevaCoT-7B-Stock"
  "fangcaotank/task-10-Qwen-Qwen2.5-7B-Instruct"
  "SeeFlock/task-12-Qwen-Qwen2.5-7B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
  "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8"
  "meta-llama/Llama-3.1-8B"
  "meta-llama/Llama-3.1-8B-Instruct"
  "ValiantLabs/Llama3.1-8B-Fireplace2"
  "RedHatAI/Llama-3.1-8B-tldr"
  "proxectonos/Llama-3.1-Carballo"
  "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"
  "gowtham58/MistLlama-0.1"
  "Xiaojian9992024/Llama3.1-8B-ExtraMix"
  "LlamaFactoryAI/Llama-3.1-8B-Instruct-cv-job-description-matching"
  "chchen/Llama-3.1-8B-Instruct-PsyCourse-fold7"
  "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic"
  "mlx-community/Llama-3.1-8B-Instruct-4bit"
  "mistralai/Mistral-7B-v0.3"
  "mistralai/Mistral-7B-Instruct-v0.3"
  "KurmaAI/AQUA-7B"
  "openfoodfacts/spellcheck-mistral-7b"
  "grimjim/Mistral-7B-Instruct-demi-merge-v0.3-7B"
  "chaymaemerhrioui/mistral-Brain_Model_ACC_Trainer"
  "RedHatAI/Mistral-7B-Instruct-v0.3-GPTQ-4bit"
  "google/gemma-2-2b"
  "google/gemma-2-2b-it"
  "rinna/gemma-2-baku-2b"
  "anakin87/gemma-2-2b-neogenesis-ita"
  "google-cloud-partnership/gemma-2-2b-it-lora-sql"
  "vonjack/gemma2-2b-merged"
  "qilowoq/gemma-2-2B-it-4Bit-GPTQ"
  "microsoft/Phi-4-mini-instruct"
  "madcows/siwon-mini-instruct-0626"
  "Pinkstack/Phi-4-mini-6b-merge"
  "SeeFlock/task-10-microsoft-Phi-4-mini-instruct"
  "iqbalamo93/Phi-4-mini-instruct-GPTQ-8bit"
  "Qwen/Qwen2.5-14B"
  "Qwen/Qwen2.5-14B-Instruct"
  "Qwen/Qwen2.5-Coder-14B"
  "oxyapi/oxy-1-small"
  "v000000/Qwen2.5-14B-Gutenberg-Instruct-Slerpeno"
  "ToastyPigeon/qwen-story-test-qlora"
  "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4"
)

# Specify the local directory to download models to.
# If left empty, the default Hugging Face cache directory will be used (~/.cache/huggingface/hub).
# Specifying a directory is recommended for easier management.
# TARGET_DIR="./models"

# Your Hugging Face Hub Token.
# If you need to download private models or models that require authentication (e.g., Llama 2, Gemma),
# provide your token here.
# You can create a token here: https://huggingface.co/settings/tokens
# Leave empty to download only public models.
HF_TOKEN="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" # e.g., "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"


# --- Script Body ---

# Check if the huggingface-cli command exists
if ! command -v huggingface-cli &> /dev/null
then
    echo "Error: 'huggingface-cli' command not found."
    echo "Please install it first using: pip install -U \"huggingface_hub[cli]\""
    exit 1
fi

echo "Hugging Face model download script started."
echo "========================================="

# If a target directory is specified, create it
# if [ -n "$TARGET_DIR" ]; then
#     echo "Target download directory: $TARGET_DIR"
#     mkdir -p "$TARGET_DIR"
# fi

# Prepare authentication arguments
TOKEN_ARG=""
if [ -n "$HF_TOKEN" ]; then
    echo "Hugging Face Token detected, will be used for authentication."
    TOKEN_ARG="--token $HF_TOKEN"
fi

# Loop through the model list and download
for model_id in "${MODELS_TO_DOWNLOAD[@]}"
do
    echo -e "\n-----------------------------------------"
    echo "Preparing to download model: $model_id"
    echo "-----------------------------------------"

    # Construct the download command
    CMD="huggingface-cli download --resume-download $model_id"

    # Add token argument
    if [ -n "$TOKEN_ARG" ]; then
        CMD="$CMD $TOKEN_ARG"
    fi

    # Add local directory argument
    # Using --local-dir-use-symlinks auto can save disk space by using symlinks where possible.
    # if [ -n "$TARGET_DIR" ]; then
    #     CMD="$CMD --local-dir \"$TARGET_DIR/$model_id\" --local-dir-use-symlinks auto"
    # fi
    
    # Print the command to be executed (for debugging)
    echo "Executing command: $CMD"
    
    # Execute the download
    eval $CMD

    # Check the exit code of the last command
    if [ $? -eq 0 ]; then
        echo "✅ Successfully downloaded model: $model_id"
    else
        echo "❌ Failed to download model: $model_id" >&2
    fi
done

echo -e "\n========================================="
echo "🎉 All specified model tasks have been processed."
echo "========================================="
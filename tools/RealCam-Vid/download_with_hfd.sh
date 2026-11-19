#!/bin/bash
# Alternative download method using hfd.sh as mentioned in the RealCam-Vid repo

DATA_ROOT="/common/users/sf895/wsg/input/RealCam-Vid"

echo "RealCam-Vid Dataset Downloader (hfd.sh method)"
echo "=============================================="
echo "This method requires:"
echo "1. HuggingFace username"
echo "2. HuggingFace access token (get from https://huggingface.co/settings/tokens)"
echo ""

# Create directory
mkdir -p $DATA_ROOT

# Download hfd.sh script
if [ ! -f "hfd.sh" ]; then
    echo "Downloading hfd.sh script..."
    wget https://gist.githubusercontent.com/padeoe/697678ab8e528b85a2a7bddafea1fa4f/raw/6891c4b02f5cf3d014c7b1523556e15d9a3dd00f/hfd.sh
    chmod u+x hfd.sh
fi

# Install aria2 if not present
if ! command -v aria2c &> /dev/null; then
    echo "aria2 not found. Please install it with: sudo apt-get install aria2"
    exit 1
fi

# Use official endpoint since we're in the US
# export HF_ENDPOINT=https://hf-mirror.com

# Get credentials
read -p "Enter your HuggingFace username: " HF_USERNAME
read -s -p "Enter your HuggingFace access token: " HF_TOKEN
echo ""

# Download dataset
echo "Starting download to $DATA_ROOT..."
./hfd.sh MuteApo/RealCam-Vid --dataset \
    --hf_username "$HF_USERNAME" --hf_token "$HF_TOKEN" \
    --tool aria2c -x 16 

# Move to target directory
mv RealCam-Vid/* $DATA_ROOT/

# Unzip files
echo "Unzipping downloaded files..."
cd $DATA_ROOT
find "zip" -type f -name "*.zip" -exec sh -c '
    for zip_file do
        echo "Extracting $zip_file..."
        unzip -q "$zip_file"
    done
' sh {} +

echo "Download completed!"
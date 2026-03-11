#!/bin/bash

DATA_DIR="./data"
ZIP_FILE="$DATA_DIR/pascal-voc-2012-dataset.zip"
EXTRACT_DIR="$DATA_DIR/pascal-voc-2012"
mkdir -p "$DATA_DIR"
echo "Downloading dataset..."
curl -L -o "$ZIP_FILE" \
https://www.kaggle.com/api/v1/datasets/download/gopalbhattrai/pascal-voc-2012-dataset
mkdir -p "$EXTRACT_DIR"
unzip -q "$ZIP_FILE" -d "$EXTRACT_DIR"
rm "$ZIP_FILE"
echo "Done! Data is located in $EXTRACT_DIR"

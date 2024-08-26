#!/bin/bash

# Change this to your root directory
ROOT_DIR="$1"
SAVE_DIR="$2"
CONFIG_PATH="$3"

# The fixed subfolder name
FIXED_SUBFOLDER="ego/depth_pcd/"

# Check if the config file exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Config file not found: $CONFIG_PATH"
    exit 1
fi

# Check if the directory exists
if [ ! -d "$SAVE_DIR" ]; then
  # The directory does not exist, creating it
  mkdir -p "$SAVE_DIR"
  echo "Directory created: $SAVE_DIR"
else
  echo "Directory already exists: $SAVE_DIR"
fi

# Check if the directory exists
if [ -d "$SAVE_DIR" ]; then
    # Remove all files in the directory
    rm -f -r $SAVE_DIR/*
else
    echo "Directory does not exist."
fi

# Iterate over each subfolder in the root directory
for dir in "$ROOT_DIR"/*; do
    if [ -d "$dir" ]; then
        # Construct the new directory path
        NEW_DIR_PATH="$dir/$FIXED_SUBFOLDER"      
        # Check if the new directory path exists and is a directory
        if [ -d "$NEW_DIR_PATH" ]; then
            # Run the command for each subfolder
            kiss_icp_pipeline --config "$CONFIG_PATH" "$NEW_DIR_PATH"
            echo "Finish odometry pipeline for: $dir"
        else
            echo "Directory not found: $NEW_DIR_PATH"
        fi
    fi
done
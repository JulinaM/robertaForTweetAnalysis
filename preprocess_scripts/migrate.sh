#!/bin/bash

# Define the root source directory and root destination directory
root_source_dir="/data2/julina/scripts/tweets/2019"
root_destination_dir="/data3/jm/tweets/2019"

# Find all JSON files in the source directory and its subdirectories
find "$root_source_dir" -type f -name "*.json" | while read source_file; do
    # Construct the destination path
    relative_path="${source_file#$root_source_dir}"
    destination_file="$root_destination_dir$relative_path"

    # Create the destination directory if it doesn't exist
    destination_dir=$(dirname "$destination_file")
    mkdir -p "$destination_dir"

    # Move the JSON file to the destination directory
    echo "Moving $source_file to $destination_file"
    mv "$source_file" "$destination_file"
done

echo "All .json files moved to $root_destination_dir"


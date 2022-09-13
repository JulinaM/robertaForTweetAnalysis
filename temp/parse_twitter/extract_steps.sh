#!/usr/bin/env bash

# Usage:
# bash extract_steps.sh [target_dir]

# cd $1
# cd test/

# delete .DS_Store all
find . -iname ".DS_Store" -delete

# manipulate day folder
for dir in `ls -d -- */`; do
    find $dir -iname "*.json" -exec cat {} >> "${dir:0:8}".json \;
    cd $dir
    # FIXME  filter '.' and '..' directory.
    find . -type d -exec rm -rf {} \;
    cd ..
done

# mv */*.json . 
rmdir *
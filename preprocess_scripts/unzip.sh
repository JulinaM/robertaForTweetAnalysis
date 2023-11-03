#!/usr/bin/env bash

# Usage:
# bash extract_steps.sh [target_dir]
#bash ../unzip.sh /data2/TwitterStreamRawData/TwitterData_2020/11/ 

src=$1
# dest=$2
# #find $1 -name "*.tar" -exec sh -c 'tar xvf {} -C $(dirname {})' \;
find $src -type f -name "*.zip" -exec sh -c 'for file do
    dir="$(basename )"
    echo "$dir"
    unzip -d "$dir" "$file"
done' sh {} +

# for dir in `ls -d -- */`; do
#     echo $dir
#     cd $dir
    
#     cd ..
# done

#find . -iname "*.json.bz2" -exec bunzip2 {} \;

# manipulate day folder
# for dir in `ls -d -- */`; do
    
#     find . -iname "*.json.bz2" -exec bunzip2 {} \;
#     find $dir -iname "*.json" -exec cat {} >> "2020_09_${dir:0:2}".json \;
#     cd $dir
#     # FIXME  filter '.' and '..' directory.
#     find . -type d -exec rm -rf {} \;
#     cd ..
# done

# # mv */*.json . 
# rmdir *
# tar xvf "$tarfile" -C "$dir"
# find $src -type f -name "*.zip" -exec sh -c 'for file do
#     dir="$(basename "$file" .zip)"
#     mkdir -p "$dir"
#     echo "created... $file"
#     unzip -d "$dir" "$file"
# done' sh {} +
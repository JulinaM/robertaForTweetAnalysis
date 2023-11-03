#!/usr/bin/env bash

# Usage:
#bash ../extract_steps_z.sh /data2/TwitterStreamRawData/TwitterData_2020/11/ 
#go to /data2/julina/scripts/tweets/2020 and run for a month

# month="09"
month=$1
# for month in `ls -d -- */`; do
    echo " ----> $month"
    cd $month
    for day in `ls -d -- */`; do
        echo ">>>>> ${day}"
        cd $day 
        find . -iname "*.json.bz2" -exec bunzip2 {} \;
        find . -iname "*.json" -exec cat {} >> ../"2020_${month}_${day:0:2}".json \;
        
        # find . -iname "*.json.gz" -exec gzip -d {} \;
        # find . -iname "*.json" -exec cat {} >> ../"${day:0:8}".json \;

        # mv "2020_${month}_${day:0:2}".json ../
        cd ..
    done
    cd ..
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
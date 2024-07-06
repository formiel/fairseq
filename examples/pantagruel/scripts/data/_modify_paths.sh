#!/bin/bash

# Check if both arguments are provided
if [ $# -ne 3 ]; then
    echo "Usage: $0 <file>"
    exit 1
fi

# Assign arguments to variables
file=$1
old_text=$2 #"/gpfsscratch/rech/ahm/umz16dj/Data"
replacement_text=$3 #"/lus/work/CT10/c1615074/tphle/Data/raw"

# Escape special characters in old_text and replacement_text
escaped_old_text=$(sed 's/[^^]/[&]/g; s/\^/\\^/g' <<< "$old_text")
escaped_replacement_text=$(sed 's/[\/&]/\\&/g' <<< "$replacement_text")

# Run sed command with arguments
sed -i "s/$escaped_old_text/$escaped_replacement_text/g" "$file"


#!/usr/bin/env bash

CAND_LINKS_DIR="/nfs1/shared/for_aniruddha/pose/cand_link_dirs"

for dir in "$CAND_LINK_DIR"/*
do
    if ! [[ -d "$dir" ]]
    then
        continue
    fi

    vid_name="${dir##*/}"
    mkdir vid_name

    ~/openpose/openpose.bin \
        --image_dir "$dir" \
        --write_json "$vid_name" \
        --display 0 \
        --render_pose 1
done

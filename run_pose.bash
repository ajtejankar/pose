#!/usr/bin/env bash

CAND_LINKS_DIR="/nfs1/shared/for_aniruddha/pose/cand_frame_links"
FORMAT="%-20s: %s\n"

out_dir="$(pwd)"
pushd ~/openpose/

for dir in "${@}"
do
    [[ -d "$dir" ]] || continue
    vid_name="$out_dir/${dir##*/}"

    printf "$FORMAT" "frame_dir" "$dir"
    printf "$FORMAT" "dump_json" "$vid_name"

    if [[ -d "$vid_name" ]]; then
        printf "$FORMAT" "skip_path" "$vid_name"
        continue
    fi

    printf "$FORMAT" "start_dir" "$dir"

    CUDA_VISIBLE_DEVICES=6,7 ~/openpose/openpose.bin \
        --image_dir "$dir" \
        --write_json "$vid_name" \
        --display 0 \
        --render_pose 1
    
    printf "$FORMAT" "done_dir" "$dir"
done

popd

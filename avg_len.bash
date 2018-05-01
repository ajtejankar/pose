#!/usr/bin/env bash

for file in "$@"
do
	duration="$(ffmpeg -i $file 2>&1 | \
		awk '/Duration/ { hh=substr($2,4,2); ss=substr($2,7,2); print (hh*60)+ss;}')";

	printf "%-60s %-20s\n" "$file" "${duration} s"
done | awk '{ tot += $2; print $0; } END { print "\nAVG " tot/NR " s"}'

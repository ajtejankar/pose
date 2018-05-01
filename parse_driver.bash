#!/usr/bin/env bash

awk </dev/stdin '
    $2 == "python" { print $0 }
'

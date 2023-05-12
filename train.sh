#!/usr/bin/env bash

python main.py --input_vars "vv_before,vv_after,vh_before,vh_after" --timestep_length 1 --ds_path data/hokkaido_japan.zarr --num_workers 4

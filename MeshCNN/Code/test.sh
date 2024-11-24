#!/usr/bin/env bash

## run the test and export collapses
CUDA_VISIBLE_DEVICES="0" python3 test.py \
--gpu_ids 0 \
--dataroot datasets/testseeds \
--name testweights \
--ncf 64 128 256 256 \
--ninput_edges 9000 \
--pool_res 7200 6300 5400 4500 \
--norm group \
--resblocks 1 \
--export_folder meshes \
--which_epoch 1 \

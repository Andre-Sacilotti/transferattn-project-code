#!/bin/bash
# ----------------------------------------------------------------------------------
# variable
data_path=../../datasets/hmdb_ucf/ucf101/RGB # depend on users
# video_in=""
feature_in=RGB-feature
input_type=frames # video | frames
structure=tsn # tsn | imagenet
num_thread=1
batch_size=128 # need to be larger than 16 for c3d
base_model=TRN # resnet101 | c3d
pretrain_weight=./stam_16.pth # depend on users (only used for C3D model)
start_class=6 # start from 1
end_class=-1 # -1: process all the categories
class_file=../TA3N/data/ucf101_splits/class_list_hmdb_ucf.txt # none | XXX/class_list_DA.txt (depend on users)

CUDA_VISIBLE_DEVICES=1 python -W ignore video2feature_trn.py --data_path $data_path \
--feature_in $feature_in --input_type $input_type --structure $structure \
--num_thread $num_thread --batch_size $batch_size --base_model $base_model --pretrain_weight $pretrain_weight \
--start_class $start_class --end_class $end_class --class_file $class_file


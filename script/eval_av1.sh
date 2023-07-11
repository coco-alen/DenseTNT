
python src/run.py \
    --core_num 36 --distributed_training 2 \
    --argoverse \
    --future_frame_num 30 \
    --do_train \
    --reuse_temp_file \
    --data_dir /home/DATA/yipin/dataset/Argoverse/motion_forecasting/train/data \
    --data_dir_for_val /home/DATA/yipin/dataset/Argoverse/motion_forecasting/val/data \
    --temp_file_dir /home/DATA/yipin/dataset/Argoverse/motion_forecasting/ \
    --output_dir /home/yipin/program/nas/nasTNT/DenseTNT/ckpt/av1/densetnt/baseline \
    --hidden_size 128 \
    --train_batch_size 64 \
    --use_map \
    --use_centerline \
    --other_params \
    semantic_lane direction l1_loss \
    goals_2D enhance_global_graph subdivide goal_scoring laneGCN point_sub_graph \
    lane_scoring complete_traj complete_traj-3 \
    --do_eval \
    --eval_params optimization MRminFDE cnt_sample=9 opti_time=0.1
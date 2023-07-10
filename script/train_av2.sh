python src/run.py \
    --core_num 36 --distributed_training 2 \
    --use_map \
    --use_centerline \
    --argoverse \
    --argoverse2 \
    --future_frame_num 60 \
    --do_train \
    --data_dir /home/DATA/yipin/dataset/Argoverse2/motion_forecasting/train/data \
    --output_dir /home/yipin/program/nas/nasTNT/DenseTNT/ckpt/av2/densetnt/1 \
    --hidden_size 128 \
    --train_batch_size 64 \
    --reuse_temp_file \
    --other_params \
    semantic_lane direction l1_loss \
    goals_2D enhance_global_graph subdivide goal_scoring laneGCN point_sub_graph \
    lane_scoring complete_traj complete_traj-3 \
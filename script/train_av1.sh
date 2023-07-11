
python -u src/run.py \
    --core_num 36 --distributed_training 2 \
    --master_port 12357 \
    --argoverse \
    --future_frame_num 30 \
    --do_train \
    --data_dir /home/DATA/yipin/dataset/Argoverse/motion_forecasting/train/data \
    --data_dir_for_val /home/DATA/yipin/dataset/Argoverse/motion_forecasting/val/data \
    --output_dir /home/yipin/program/nas/nasTNT/DenseTNT/ckpt/av1/densetnt/bigSize \
    --hidden_size 512 \
    --train_batch_size 64 \
    --num_train_epochs 16 \
    --weight_decay 0.01 \
    --learning_rate 0.001 \
    --sub_graph_depth 6 \
    --use_map \
    --use_centerline \
    --reuse_temp_file \
    --temp_file_dir /home/DATA/yipin/dataset/Argoverse/motion_forecasting/ \
    --other_params \
    semantic_lane direction l1_loss \
    goals_2D enhance_global_graph subdivide goal_scoring laneGCN point_sub_graph \
    lane_scoring complete_traj complete_traj-3 \
# ---------- UCF-HMDB Experiments ----------

# UCF -> HMDB, using I3D features
CUDA_VISIBLE_DEVICES=0 python main.py --dataset UCF_HMDB_I3D --source ucf --target hmdb --adv_coeff 1 --queue_size 512 --IB_weight 0.001 --batch_size 32 --sampled_frame 53 --use_ib_loss --use_adv_loss --use_MDTA --random_seed 42

# HMDB -> UCF, using I3D features
CUDA_VISIBLE_DEVICES=0 python main.py --dataset UCF_HMDB_I3D --source hmdb --target ucf --adv_coeff 0.5 --queue_size 512 --IB_weight 0.001 --batch_size 32 --sampled_frame 53 --use_ib_loss --use_adv_loss --use_MDTA --random_seed 5

# UCF -> HMDB, using STAM features
CUDA_VISIBLE_DEVICES=0 python main.py --dataset UCF_HMDB_STAM --source ucf --target hmdb --adv_coeff 1 --queue_size 512 --IB_weight 0.001 --batch_size 32 --sampled_frame 53 --use_ib_loss --use_adv_loss --use_MDTA --random_seed 42

# HMDB -> UCF, using STAM features
CUDA_VISIBLE_DEVICES=0 python main.py --dataset UCF_HMDB_STAM --source hmdb --target ucf --adv_coeff 0.5 --queue_size 512 --IB_weight 0.001 --batch_size 32 --sampled_frame 53 --use_ib_loss --use_adv_loss --use_MDTA --random_seed 5

# ---------- Kinetics - NEC-Drone Experiments ----------

# K -> N, using STAM Features
CUDA_VISIBLE_DEVICES=0 python main.py --dataset K_NEC --source kinetics --target nec --adv_coeff 0.5 --queue_size 512 --IB_weight 0.025 --batch_size 64 --sampled_frame 53 --use_ib_loss  --use_adv_loss --use_MDTA --random_seed 42
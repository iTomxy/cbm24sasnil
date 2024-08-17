set -e

echo 1st stage: initial training
lp=log/1st
CUDA_VISIBLE_DEVICES=1 python stage1.py totalseg-spineLSpelvic-small --rm_old_ckpt --log_path $lp --data_root ~/data \
    --batch_size 32 --lr 5e-5 --iter 200 --val_freq 5 --partial spine --window --obviousness 500
python test.py totalseg-spineLSpelvic-small $lp/ckpt-140.pth

echo 2nd stage: label correction training
lp=log/2nd
tw=log/1st/ckpt-140.pth
python stage2.py totalseg-spineLSpelvic-small --rm_old_ckpt --log_path $lp --data_root ~/data \
    --iter 10000 --val_freq 50 --teacher_weight $tw --ema --ema_start 140 --ema_freq 1 --ema_momentum 0.95
python test.py totalseg-spineLSpelvic-small $lp/best_val.pth

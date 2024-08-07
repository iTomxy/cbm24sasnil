set -e

echo 1st stage: initial training
lp=log/1st
CUDA_VISIBLE_DEVICES=1 python init_train.py totalseg-spineLSpelvic-small --rm_old_ckpt --log_path $lp --data_root ~/data \
    --batch_size 32 --lr 5e-5 --iter 200 --val_freq 5 --partial spine --window --obviousness 500


# echo \[`date`\] `whoami` @ `hostname` : `realpath $0` | mail -s "cmd done" tianyou.liang@student.uts.edu.au

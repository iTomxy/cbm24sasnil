set -e
trap 'echo \[`date`\] `whoami`@`hostname`:`realpath $0`, $LINENO | mail -s "cmd error" tianyou.liang@student.uts.edu.au' ERR
trap 'echo \[`date`\] `whoami`@`hostname`:`realpath $0`, $LINENO | mail -s "cmd interrupted" tianyou.liang@student.uts.edu.au' TERM HUP

# select 1st stage model by validation dice socre
best_i=-1 # best iteration
best_v=0  # best dice score
pick_best() {
    # $1: validation dynamic json log file
    res=$(awk -F '"iter": |"dice": ' '{if ($2 && $3) {split($2, iter, ","); split($3, dice, ","); print iter[1], dice[1] * 10000}}' $1 | \
        sort -k 2 -nr | head -n 1)
    best_i=`echo $res | awk '{print $1}'`
    best_v=0.`echo $res | awk '{print $2}'`
}

dset=totalseg-spineLSpelvic-small
data_root=$HOME/data/totalsegmentator

#
# totalseg-spineLSpelvic-small
#

echo 1st stage: initial training
lp=log/$dset/1st
python stage1.py $dset --rm_old_ckpt --log_path $lp --data_root $data_root \
    --batch_size 32 --lr 5e-5 --iter 200 --val_freq 5 --window --obviousness 500
pick_best $lp/dynamics-val.json
echo $best_i, $best_v
python test.py $dset $lp/ckpt-${best_i}.pth --data_root $data_root

# echo 2nd stage: label correction training
# lp=log/2nd
# tw=log/1st/ckpt-140.pth
# python stage2.py totalseg-spineLSpelvic-small --rm_old_ckpt --log_path $lp --data_root ~/data \
#     --iter 10000 --val_freq 50 --teacher_weight $tw --ema --ema_start 140 --ema_freq 1 --ema_momentum 0.95
# python test.py totalseg-spineLSpelvic-small $lp/best_val.pth


# echo 2nd stage: label correction training, use T pred as pseudo at the beginning
# lp=log/2nd-cl140
# tw=log/1st/ckpt-140.pth
# python stage2.py totalseg-spineLSpelvic-small --rm_old_ckpt --log_path $lp --data_root ~/data \
#     --iter 10000 --val_freq 50 --teacher_weight $tw --ema --ema_start 140 --ema_freq 1 --ema_momentum 0.95 --cl_pseudo_start 140


echo 2nd stage: label correction training, continue training from 1st ckpt
lp=log/$dset/2nd-re_stu
tw=log/$dset/1st/ckpt-${best_i}.pth
python stage2.py $dset --rm_old_ckpt --log_path $lp --data_root $data_root \
    --iter 10000 --val_freq 50 --teacher_weight $tw --ema --ema_start 0 --ema_freq 1 --ema_momentum 0.95 --cl_pseudo_start 0 --resume_student
python test.py $dset $lp/best_val.pth --data_root $data_root

#
# totalseg-spineCshoulder-small
#

# echo 1st stage: initial training
# lp1=log/1st-shoulder
# python stage1.py totalseg-spineCshoulder-small --rm_old_ckpt --log_path $lp1 --data_root ~/data \
#     --batch_size 32 --lr 5e-5 --iter 200 --val_freq 5 --partial spine --window --obviousness 500
# pick_best $lp1/dynamics-val.json
# echo $best_i, $best_v
# python test.py totalseg-spineCshoulder-small $lp1/ckpt-${best_i}.pth

# echo 2nd stage: label correction training, continue training from 1st ckpt
# tw=$lp1/ckpt-${best_i}.pth
# lp2=log/2nd-shoulder
# python stage2.py totalseg-spineCshoulder-small --rm_old_ckpt --log_path $lp2 --data_root ~/data \
#     --iter 10000 --val_freq 50 --teacher_weight $tw --ema --ema_start 0 --ema_freq 1 --ema_momentum 0.95 --cl_pseudo_start 0 --resume_student
# python test.py totalseg-spineCshoulder-small $lp2/best_val.pth


echo \[`date`\] `whoami` @ `hostname` : `realpath $0` | mail -s "cmd done" tianyou.liang@student.uts.edu.au

import argparse, os, os.path as osp, json, shutil, itertools, functools, math
from collections import defaultdict
import numpy as np
from scipy.optimize import curve_fit
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
import monai
from monai.networks.utils import one_hot
from data import *
from util import *
from modules import *
from evaluation import *
from config import stage2_args, TOTALSEG_CLS_SET
from stage1 import vis_pred

"""
Label correction training
"""

def pseudo_by_fn_estim(partial_bin_lab, pred_sm):
    """calculate (binary) pseudo-label by estimating FN
    Ref: (cbm'24) Unsupervised domain adaptation for histopathology image segmentation with incomplete labels
    Input:
        partial_bin_lab: [h, w], in {0, 1}
        pred_sm: [c, h, w], in [0, 1]
    Output:
        pseudo: [h, w], in {0, 1}
    """
    nc, h, w = pred_sm.shape
    gamma = np.zeros([nc], dtype=np.float32) + 2 # 2 > 1, represents impossible probability threshold
    for ic in range(nc):
        _msk = (ic == partial_bin_lab)
        if _msk.any():
            gamma[ic] = pred_sm[ic][_msk].mean()

    C = np.zeros([2, 2], dtype=np.float32)
    for ic, jc in itertools.product(range(nc), range(nc)):
        C[ic][jc] = ((ic == partial_bin_lab) & (pred_sm[jc] >= gamma[jc])).sum()

    _c_denom = C.sum(1, keepdims=True)
    C_norm_j = C / np.where(_c_denom > 0, _c_denom, 1)
    X_card_i = np.asarray([
        (ic == partial_bin_lab).sum() for ic in range(nc)
    ], dtype=np.float32)[np.newaxis, :] # [1, c]
    Q_numer = C_norm_j * X_card_i
    # _q_denom = Q_numer.sum(0, keepdims=True)
    # Q = Q_numer / np.where(_q_denom > 0, _q_denom, 1)
    _q_denom = Q_numer.sum()
    Q = Q_numer / (_q_denom if _q_denom > 0 else 1)

    n_fn = int(Q[0][1] * h * w) # estimated #FN pixels
    asc_p0 = np.sort(pred_sm[0, 0 == partial_bin_lab].flatten())
    thres = asc_p0[n_fn - 1]
    pseudo = np.zeros_like(partial_bin_lab)
    _where = np.where((0 == partial_bin_lab) & (pred_sm[0] <= thres))
    _where = [_p[:n_fn] for _p in _where] # ensure FN pixel number
    pseudo[_where] = 1
    pseudo = (partial_bin_lab + pseudo > 0).astype(np.uint8)
    return pseudo


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build teacher & student with the same old args
    with open(osp.join(osp.dirname(args.teacher_weight), "config.json"), "r") as f:
        old_args = argparse.Namespace(**json.load(f))
    teacher = build_model(old_args).to(device)
    init_teacher_ckpt = torch.load(args.teacher_weight)
    teacher.load_state_dict(init_teacher_ckpt["model"])
    teacher.eval()
    model = build_model(old_args).to(device)
    if args.resume_student:
        model.load_state_dict(init_teacher_ckpt["model"])
    # update args
    old_args.__dict__.update(args.__dict__) # new overwrites old
    args = old_args

    train_npzs = get_bone_data_files(args.dataset, "training", args.data_root)
    val_img_niis, val_lab_niis, _, _ = get_bone_nii_list(args.dataset, "training", args.data_root)

    # label binarisation function
    bin_fn_full = binarise_totalseg_label
    if args.partial:
        assert args.partial in TOTALSEG_CLS_SET, args.partial
        bin_fn_train = functools.partial(binarise_totalseg_label, coi=TOTALSEG_CLS_SET[args.partial])
    else:
        bin_fn_train = bin_fn_full

    train_trans1, train_trans2, train_trans3 = weak_trfm2(args)
    val_trans = val_trfm(args)

    train_ds = BoneDataset(train_npzs, train_trans1, bin_fn=bin_fn_train,
        window=args.window, window_level=args.window_level, window_width=args.window_width)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())
    train_iter = infiter(train_loader)

    criterion_dice = monai.losses.DiceLoss(softmax=True, reduction='none')
    criterion_focal = monai.losses.FocalLoss(use_softmax=True, reduction='none')
    optimizer = torch.optim.AdamW([
        {"params": model.unet.parameters(), 'lr': args.lr * args.backbone_lr_coef},
        {'params': model.seg_heads.parameters()},
    ], lr=args.lr)
    scheduler = None
    if len(args.decay_steps) > 0:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.decay_steps)

    dynamic_f = osp.join(args.log_path, "dynamics.json")
    writer = SummaryWriter(log_dir=args.log_path)
    os.makedirs(args.log_path, exist_ok=True)


    def val(i_it, loss):
        # validate (use PARTIAL label -> bin_fn_train)
        res, _ = test_3d(args, model, args.dataset, "training", val_trans, bin_fn=bin_fn_train)
        # write json
        with open(dynamic_f, "a") as f:
            log = {"iter": i_it, "loss": loss}
            log.update(res)
            f.write(json.dumps(log) + os.linesep)
        # write tensorboard
        for metr, v in res.items():
            if metr.endswith("_cw"):
                continue
            writer.add_scalar(metr, v, i_it)
        # visualisation (use FULL label -> bin_fn_full)
        i_nii = np.random.choice(len(val_img_niis))
        val_ds = BoneDatasetNii(val_img_niis[i_nii], val_lab_niis[i_nii], args.slice_axis, val_trans, bin_fn_full,
            window=args.window, window_level=args.window_level, window_width=args.window_width)
        loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False, pin_memory=torch.cuda.is_available())
        vis_pred(args, [model, teacher], loader, osp.join(args.log_path, "vis", str(i_it)))
        for _old_it in range(i_it - 5, max(0, i_it - 200), -1):
            if osp.isdir(osp.join(args.log_path, "vis", str(_old_it))) and _old_it % 200 != 0:
                shutil.rmtree(osp.join(args.log_path, "vis", str(_old_it)))
        # update best dice
        return res["dice"]


    start_iter = 0
    best_dice = -1
    if args.resume:
        assert os.path.isfile(args.resume), args.resume
        ckpt = torch.load(args.resume)
        model.load_state_dict(ckpt["model"])
        teacher.load_state_dict(ckpt["teacher"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_iter = ckpt["iter"] + 1
        best_dice = ckpt["best_dice"]
        print("Resume:", args.resume, ", start from:", start_iter)
    else:
        assert not os.path.isfile(osp.join(args.log_path, "config.json"))
        with open(os.path.join(args.log_path, "config.json"), "w") as f:
            json.dump(args.__dict__, f, indent=1)

        assert not osp.isfile(dynamic_f), dynamic_f
        with open(dynamic_f, "w") as f:
            f.write(json.dumps(args.__dict__) + os.linesep)

        # val before training
        if not args.debug:
            best_dice = val(0, 1)

    # train
    max_entropy = math.log(args.n_classes)
    tup_cntdn = args.ema_freq # Teacher UPdate CouNTDowN
    model.train()
    for i_iter in range(start_iter, args.iter):
        batch_data = next(train_iter)
        labels_bin_par = batch_data["label"]
        inputs = batch_data["image"]

        # infer pseudo-label
        with torch.no_grad():
            _, tea_pred_list = teacher(inputs.to(device))
            tea_preds = tea_pred_list[0]
        # infer pseudo-labels by FN estimation
        if i_iter >= args.cl_pseudo_start:
            tea_preds_sm = tea_preds.softmax(1).cpu().numpy() # [bs, c, h, w]
            pseudos = np.vstack([
                pseudo_by_fn_estim(lab_bp, pred_sm)[np.newaxis]
                for lab_bp, pred_sm in zip(labels_bin_par[:, 0].numpy(), tea_preds_sm)
            ])
        else:
            # Before using confidence learning based pseudo labels,
            # use teacher prediction as pseudo labels.
            pseudos = tea_preds.argmax(1).cpu().numpy()
        # aggregate with given partial label
        pseudos = np.vstack([
            aggregate_label(lab_bp, plab, args.pseudo_agg_mode)[np.newaxis]
            for lab_bp, plab in zip(labels_bin_par[:, 0].numpy(), pseudos)
        ])
        pseudo_labels = torch.from_numpy(pseudos).unsqueeze(1)

        # spatial transforms
        _res = train_trans2({"image": inputs, "label": labels_bin_par, "pseudo": pseudo_labels})
        s_inputs = _res["image"]
        labels_bin_par_sp = _res["label"]
        pseudo_labels = _res["pseudo"]

        pseudo_labels_oh = one_hot(pseudo_labels, num_classes=args.n_classes, dim=1).to(device)

        # pixel transforms to student input
        s_inputs = train_trans3(s_inputs)

        optimizer.zero_grad()
        feats, pred_list = model(s_inputs.to(device))
        preds = pred_list[0]

        loss_dice = criterion_dice(preds, pseudo_labels_oh)
        loss_focal = criterion_focal(preds, pseudo_labels_oh)
        # print(loss_dice.size(), loss_focal.size()) # [bs, c, 1, 1], [bs, c, h, w]

        # manual sample-wise reduction
        loss_dice = loss_dice.mean((1, 2, 3)) # [bs]
        loss_focal = loss_focal.mean((1, 2, 3))

        # sample weight
        preds_sm = preds.detach().softmax(1) # [bs, c, h, w]
        confid, _ = preds_sm.max(1) # [bs, h, w]
        ent = 1 - (- preds_sm * preds_sm.log()).sum(1) / max_entropy
        weight = (confid * ent).mean((1, 2)) # [bs]
        # weight losses
        loss_dice = loss_dice.dot(weight)
        loss_focal = loss_focal.dot(weight)

        loss = 0.5 * (loss_dice + loss_focal)
        loss.backward()
        optimizer.step()
        writer.add_scalar("loss_dice", loss_dice.item(), i_iter + 1)
        writer.add_scalar("loss_focal", loss_focal.item(), i_iter + 1)
        writer.add_scalar("loss", loss.item(), i_iter + 1)

        # vis batch
        vis_batch_dir = osp.join(args.log_path, "vis-batch", str(i_iter))
        os.makedirs(vis_batch_dir, exist_ok=True)
        for i, (img, lab_bp, pred_t, lab_ps, simg, lab_bp_sp, pred) in enumerate(zip(
            inputs.numpy(), labels_bin_par.numpy(), tea_preds.argmax(1).cpu().numpy(),
            pseudo_labels.numpy(),
            s_inputs.numpy(), labels_bin_par_sp.numpy(), preds.detach().argmax(1).cpu().numpy(),
        )):
            comb = [np.tile(np.clip(im * 255, 0, 255).astype(np.uint8)[:, :, np.newaxis], (1, 1, 3)) for im in img]
            comb.extend([
                np.asarray(color_seg(lab_bp[0])),
                bin_seg_err_map(y=lab_bp[0], pred=pred_t),
            ])
            comb.extend([np.tile(np.clip(im * 255, 0, 255).astype(np.uint8)[:, :, np.newaxis], (1, 1, 3)) for im in simg])
            comb.extend([
                np.asarray(color_seg(lab_bp_sp[0])),
                bin_seg_err_map(y=lab_bp_sp[0], pred=lab_ps[0]),
                bin_seg_err_map(y=lab_bp_sp[0], pred=pred),
            ])
            Image.fromarray(compact_image_grid(comb)).save(osp.join(vis_batch_dir, f"{i}.png"))
        # rm old batch vis
        if osp.isdir(osp.join(args.log_path, "vis-batch", str(i_iter - 10))):
            shutil.rmtree(osp.join(args.log_path, "vis-batch", str(i_iter - 10)))

        if args.ema and i_iter >= args.ema_start:
            tup_cntdn -= 1
            if 0 == tup_cntdn:
                tup_cntdn = args.ema_freq # reset
                with torch.no_grad():
                    for teacher_param, student_param in zip(teacher.parameters(), model.parameters()):
                        teacher_param.data = args.ema_momentum * teacher_param.data + (1 - args.ema_momentum) * student_param.data

        ckpt = {
            "iter": i_iter,
            "model": model.state_dict(),
            "teacher": teacher.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_dice": best_dice, # currently old best dice
        }
        if scheduler is not None:
            scheduler.step()
            ckpt["scheduler"] = scheduler.state_dict()
        if (i_iter + 1) % args.val_freq == 0 or i_iter + 1 == args.iter or args.debug:
            val_dice = val(i_iter + 1, loss.item())
            if val_dice > best_dice:
                ckpt["best_dice"] = best_dice = val_dice # update
                torch.save(ckpt, osp.join(args.log_path, f"best_val.pth"))

        torch.save(ckpt, osp.join(args.log_path, f"ckpt-{i_iter}.pth"))
        if args.rm_old_ckpt:
            if osp.isfile(osp.join(args.log_path, f"ckpt-{i_iter - 1}.pth")):
                os.remove(osp.join(args.log_path, f"ckpt-{i_iter - 1}.pth"))

        if args.debug:
            break


if "__main__" == __name__:
    args = stage2_args()

    assert args.val_freq > 0
    assert args.teacher_weight and osp.isfile(args.teacher_weight)

    train(args)

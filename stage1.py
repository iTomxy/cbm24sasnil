import argparse, os, os.path as osp, json, shutil, functools, glob
from collections import defaultdict
import numpy as np
from scipy.optimize import curve_fit
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
import monai
from monai.networks.utils import one_hot
from data.aug import weak_trfm, val_trfm
from data import ts_spinelspelvic, pengwin, ctpelvic1k
from util import *
from modules import *
from evaluation import *
from config import stage1_args

"""
Initial training based on early learning
"""

@torch.no_grad()
def vis_pred(args, model_list, loader, log_path):
    """loader: BoneDatasetNii"""
    if osp.isdir(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path)#, exist_ok=True)
    assert isinstance(model_list, (list, tuple))
    device_list = []
    for model in model_list:
        model.eval()
        device_list.append(next(model.parameters()).device)
    cnt = 0
    max_ent = np.log(args.n_classes)
    for batch_data in loader:
        images = batch_data["image"]
        preds_list, preds_sm_list = [], []
        for model, device in zip(model_list, device_list):
            _, preds = model(images.to(device))
            preds_list.append(preds[0].argmax(1).cpu().numpy())  # [bs, h, w]
            preds_sm_list.append(preds[0].softmax(1).cpu().numpy())  # [bs, c, h, w]
        images = images.cpu().numpy()
        labels = batch_data['label'].numpy()
        for i in range(images.shape[0]):
            comb_list = []
            for _im in images[i]:
                comb_list.append(np.tile(np.clip(_im * 255, 0, 255).astype(np.uint8)[:, :, np.newaxis], (1, 1, 3)))
            comb_list.append(np.asarray(color_seg(labels[i, 0])))
            for preds, preds_sm in zip(preds_list, preds_sm_list):
                comb_list.append(np.asarray(color_seg(preds[i])))
                comb_list.append(bin_seg_err_map(y=labels[i, 0], pred=preds[i]))
                # uncertainty
                ent = (- preds_sm[i] * np.log(preds_sm[i])).sum(0) / max_ent
                ent_map = np.tile(np.clip(ent[:, :, np.newaxis] * 255, 0, 255).astype(np.uint8), (1, 1, 3))
                comb_list.append(ent_map)
            comb = compact_image_grid(comb_list)
            Image.fromarray(comb).save(osp.join(log_path, f"{cnt}.png"))
            cnt += 1
            if args.debug:
                break
        if args.debug:
            break


def train(args):
    if "totalseg-spineLSpelvic-small" == args.dataset:
        data_pkg = ts_spinelspelvic
    elif "pengwin" == args.dataset:
        data_pkg = pengwin
    elif "ctpelvic1k" == args.dataset:
        data_pkg = ctpelvic1k

    # train_npzs = get_bone_data_files(args.dataset, "training", args.data_root)
    # val_img_niis, val_lab_niis, _, _ = get_bone_nii_list(args.dataset, "training", args.data_root)
    train_vids = data_pkg.get_split_vids("training")
    val_vids = data_pkg.get_split_vids("validation")

    # # label binarisation function
    # bin_fn_full = binarise_totalseg_label
    # if args.partial:
    #     assert args.partial in TOTALSEG_CLS_SET, args.partial
    #     bin_fn_train = functools.partial(binarise_totalseg_label, coi=TOTALSEG_CLS_SET[args.partial])
    # else:
    #     bin_fn_train = bin_fn_full

    train_trans1, train_trans2 = weak_trfm(args)
    val_trans = val_trfm(args)

    # train_ds = BoneDataset(train_npzs, train_trans1, bin_fn=bin_fn_train,
    #     window=args.window, window_level=args.window_level, window_width=args.window_width)
    train_ds = data_pkg.SliceDataset("training", "partial", transform=train_trans1,
        window=args.window, window_level=args.window_level, window_width=args.window_width, data_root=args.data_root)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())
    train_iter = infiter(train_loader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args).to(device)

    criterion_dice = monai.losses.DiceLoss(softmax=True, reduction='none')
    criterion_focal = monai.losses.FocalLoss(use_softmax=True, reduction='none')
    optimizer = torch.optim.AdamW([
        {"params": model.unet.parameters(), 'lr': args.lr * args.backbone_lr_coef},
        {'params': model.seg_heads.parameters()},
    ], lr=args.lr)
    scheduler = None
    if len(args.decay_steps) > 0:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.decay_steps)

    dynamic_f = osp.join(args.log_path, "dynamics.json") # dynamic of training set -> draw curve
    dynamic_val_f = osp.join(args.log_path, "dynamics-val.json") # dynamic on validation set
    writer = SummaryWriter(log_dir=args.log_path)
    os.makedirs(args.log_path, exist_ok=True)
    with open(os.path.join(args.log_path, "config.json"), "w") as f:
        json.dump(args.__dict__, f, indent=1)


    def val(i_it, loss):
        # validate (use PARTIAL label -> bin_fn_train)
        # res, _ = test_3d(args, model, args.dataset, "training", val_trans, bin_fn=bin_fn_train)
        res, _ = test_3d(args, model, data_pkg.iter_vol_dataset(
            train_vids, "partial", val_trans, window=args.window, window_level=args.window_level, window_width=args.window_width,
            data_root=args.data_root
        ))
        # write json
        with open(dynamic_f, "a") as f:
            f.write(json.dumps({"iter": i_it, "loss": loss, "metrics": res}) + os.linesep)
        # write tensorboard
        for metr, v in res.items():
            if metr.endswith("_cw"):
                continue
            writer.add_scalar(metr, v, i_it)

        # visualisation (use FULL label -> bin_fn_full)
        # i_nii = np.random.choice(len(val_img_niis))
        # val_ds = BoneDatasetNii(val_img_niis[i_nii], val_lab_niis[i_nii], args.slice_axis, val_trans, bin_fn_full,
        #     window=args.window, window_level=args.window_level, window_width=args.window_width)
        val_ds = data_pkg.VolumeDataset(
            np.random.choice(val_vids), "full", val_trans,
            window=args.window, window_level=args.window_level, window_width=args.window_width,
            data_root=args.data_root)
        loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False, pin_memory=torch.cuda.is_available())
        vis_pred(args, [model], loader, osp.join(args.log_path, "vis", str(i_it)))

        # res_val, _ = test_3d(args, model, args.dataset, "validation", val_trans, bin_fn=bin_fn_full) # val with full label
        res_val, _ = test_3d(args, model, data_pkg.iter_vol_dataset( # val with full label
            val_vids, "full", val_trans, window=args.window, window_level=args.window_level, window_width=args.window_width,
            data_root=args.data_root
        ))
        with open(dynamic_val_f, "a") as f:
            f.write(json.dumps({"iter": i_it, "loss": loss, "metrics": res_val}) + os.linesep)


    start_iter = 0
    if args.resume:
        assert os.path.isfile(args.resume), args.resume
        ckpt = torch.load(args.resume)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_iter = ckpt["iter"] + 1
        print("Resume:", args.resume, ", start from:", start_iter)
    else:
        for dyf in (dynamic_f, dynamic_val_f):
            assert not osp.isfile(dyf), dyf
            with open(dyf, "w") as f:
                f.write(json.dumps(args.__dict__) + os.linesep)

        # val before training
        if not args.debug:
            val(0, 1)

    # train
    model.train()
    for i_iter in range(start_iter, args.iter):
        batch_data = next(train_iter)
        labels = batch_data["label"].to(device)
        inputs = batch_data["image"]#.to(device)

        # pixel transforms
        if inputs.size(1) > 1:
            inputs = torch.cat([train_trans2(inputs[:, :1]), inputs[:, 1:]], dim=1)
        else:
            inputs = train_trans2(inputs)

        labels_oh = one_hot(labels, num_classes=args.n_classes, dim=1)

        optimizer.zero_grad()
        feats, pred_list = model(inputs.to(device))
        loss_dice = criterion_dice(pred_list[0], labels_oh)
        loss_focal = criterion_focal(pred_list[0], labels_oh)
        # print(loss_dice.size(), loss_focal.size()) # [bs, c, 1, 1], [bs, c, h, w]

        # manual reduction
        loss_dice = loss_dice.mean((1, 2, 3)) # [bs]

        if args.topk_focal > 0:
            loss_focal = loss_focal.mean(dim=1).reshape(loss_focal.size(0), -1) # -> [bs, h*w]
            _, topk_idx = loss_focal.topk(k=int(loss_focal.size(1) * args.topk_focal), dim=1, largest=False, sorted=True)
            loss_focal = loss_focal.gather(1, topk_idx).mean(1) # [bs]
        else:
            loss_focal = loss_focal.mean((1, 2, 3))

        # train with samples with obvious bone
        if args.obviousness > 0:
            sample_weight = ((labels > 0).sum((1, 2, 3)) > args.obviousness).float() # [bs]
            assert sample_weight.size(0) == labels.size(0)
            loss_dice = sample_weight.dot(loss_dice)
            loss_focal = sample_weight.dot(loss_focal)
        else:
            loss_dice = loss_dice.mean()
            loss_focal = loss_focal.mean()
        writer.add_scalar("loss_dice", loss_dice.item(), i_iter + 1)
        writer.add_scalar("loss_focal", loss_focal.item(), i_iter + 1)

        loss = 0.5 * (loss_dice + loss_focal)
        loss.backward()
        optimizer.step()
        writer.add_scalar("loss", loss.item(), i_iter + 1)

        if (i_iter + 1) % args.val_freq == 0 or i_iter + 1 == args.iter or args.debug:
            val(i_iter + 1, loss.item())
            model.train()

        sd = {
            "iter": i_iter,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        if scheduler is not None:
            scheduler.step()
            sd["scheduler"] = scheduler.state_dict()
        torch.save(sd, osp.join(args.log_path, f"ckpt-{i_iter + 1}.pth"))
        if args.rm_old_ckpt and i_iter % args.val_freq != 0:
            if osp.isfile(osp.join(args.log_path, f"ckpt-{i_iter}.pth")):
                os.remove(osp.join(args.log_path, f"ckpt-{i_iter}.pth"))

        if args.debug:
            break


def exponential(x, a, b, c):
    """curve form to fit metric dynamics"""
    return a * (1 - np.exp(- b * (x ** c)))


def d_exponential(x, a, b, c):
    return a * b * c * np.exp(- b * (x ** c)) * (x ** (c - 1))


def analyse_dynamic(args):
    """draw dynamics, fit curve, calculate & save derivative on each val point"""
    dynamic_f = osp.join(args.log_path, "dynamics.json")
    assert osp.isfile(dynamic_f), dynamic_f
    it_list = []
    record = defaultdict(list)
    with open(dynamic_f, "r") as f:
        for i, ln in enumerate(f):
            if 0 == i:
                continue
            jd = json.loads(ln.strip())
            it_list.append(jd["iter"])
            for k, v in jd["metrics"].items():
                if not k.endswith("_cw") and not k.startswith("empty_"):
                    record[k].append(v)

    deriv = defaultdict(list)
    deriv["iter"] = it_list

    fig, ax = plt.subplots(3, 3, figsize=(15, 10))
    for i, (k, v_list) in enumerate(record.items()):
        r, c = i // 3, i % 3
        lns = ax[r][c].plot(it_list, v_list, label=k)
        if k in ("dice", "iou", "sensitivity", "specificity", "accuracy"):
            # ref: https://github.com/Kangningthu/ADELE/blob/master/train.py#L161
            popt, pcov = curve_fit(exponential, it_list, v_list,
                p0=(1, 1, 1), method='trf', sigma=np.geomspace(1, .1, len(v_list)), absolute_sigma=True, bounds=([0, 0, 0], [1, 1, np.inf]))
            # print("found coef:", *popt)
            _ln = ax[r][c].plot(it_list, exponential(np.asarray(it_list), *popt), label="fit")
            lns = lns + _ln
            # calculate derivatives
            deriv["param_"+k] = popt.tolist()
            for it in it_list:
                deriv[k].append(d_exponential(max(it, 1), *popt))
            # draw derivatives
            zax = ax[r][c].twinx()
            _ln = zax.plot(it_list, deriv[k], label="d", c='g')
            lns = lns + _ln
            zax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
        ax[r][c].legend(lns, [l.get_label() for l in lns], fancybox=True, framealpha=0)
        ax[r][c].set_xlabel("iter")
    fig.tight_layout()
    fig.savefig(osp.join(args.log_path, "dynamics.png"), bbox_inches='tight')
    plt.close(fig)

    with open(osp.join(args.log_path, "derivatives.json"), "w") as f:
        json.dump(deriv, f, indent=1)


def find_t0(args):
    with open(osp.join(args.log_path, "derivatives.json"), "r") as f:
        deriv = json.load(f)

    it_list = deriv["iter"]
    d_list = np.abs(deriv[args.t0_metr])

    # substitute `2k` in equation (1) with the (1st) iteration with largest derivative
    argmax_d = np.argmax(d_list) # 2k
    max_d = d_list[argmax_d] # f'(2k)

    for it, d in zip(it_list, d_list):
        if it < argmax_d:
            continue
        if d < max_d * (1 - args.t0_thres): # deformation of equation (1)
            print("t0 =", it, "under", args.t0_metr)
            with open(osp.join(args.log_path, f"t0-{args.t0_metr}.json"), "w") as f:
                json.dump({"metric": args.t0_metr, "t0": it}, f, indent=1)
            return

    raise Exception("t0 not found, consider changing t0_thres or training longer.")


if "__main__" == __name__:
    args = stage1_args()

    # assert args.partial and args.partial != "bone"
    assert args.window
    assert args.val_freq > 0

    assert len(args.image_size) in (1, 2)
    if len(args.image_size) == 1:
        args.image_size = args.image_size * 2

    if not args.resume and args.auto_resume:
        # auto-resume: find latest checkpoint when not specified
        _max_it = -1
        for f in glob.iglob(os.path.join(args.log_path, "ckpt-*.pth")):
            _it = int(os.path.basename(f)[5: -4])
            _max_it = max(_max_it, _it)

        if _max_it > 0:
            args.resume = os.path.join(args.log_path, "ckpt-{}.pth".format(_max_it))

    # back-up codes
    if not args.resume and not args.debug:
        backup_files(os.path.join(args.log_path, "backup_code"), white_list=["*.py", "*.sh", "datalist/*.json"], black_list=["log/*", ".ipynb_checkpoints/*"])

    train(args)
    analyse_dynamic(args)
    find_t0(args)

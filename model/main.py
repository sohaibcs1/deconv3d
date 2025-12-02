# deconv3d_main.py
import os, time, csv, random, argparse, json
from pathlib import Path
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# plots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 👉 import model + _to_3tuple from separate file
from deconv3d_model import TinyUNETR3D, _to_3tuple

# ----------------------------
# Utilities
# ----------------------------
def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True

def _expand(p):
    return os.path.abspath(os.path.expanduser(str(p)))

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def _append_row_csv(path, row_dict, header):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerow(row_dict)

def _plot_curves(out_dir, history):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # losses
    plt.figure()
    plt.plot(history["epoch"], history["train_loss"], label="train")
    plt.plot(history["epoch"], history["val_loss"], label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(str(Path(out_dir) / "loss_curve.png"), dpi=150); plt.close()

    # metrics
    for k in ["MAE","RMSE","PSNR","SSIM","EdgeSSIM","NCC"]:
        plt.figure()
        plt.plot(history["epoch"], history[k], label=k)
        plt.xlabel("epoch"); plt.ylabel(k); plt.legend(); plt.grid(True)
        plt.tight_layout(); plt.savefig(str(Path(out_dir) / f"{k}_curve.png"), dpi=150); plt.close()

# ----------------------------
# Data
# ----------------------------
def robust_norm(vol, p_lo=0.1, p_hi=99.9, eps=1e-6):
    lo = np.percentile(vol, p_lo)
    hi = np.percentile(vol, p_hi)
    if not np.isfinite(lo):
        lo = float(np.nanmin(vol))
    if not np.isfinite(hi):
        hi = float(np.nanmax(vol))
    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi):
        hi = 1.0
    if hi - lo < 1e-8:
        return np.zeros_like(vol, dtype=np.float32)
    vol = np.clip(vol, lo, hi)
    vol = (vol - lo) / (hi - lo + eps)
    return vol.astype(np.float32)

class PairedPatchDataset(Dataset):
    def __init__(self, pairs, patch_size=(64,64,64),
                 patches_per_volume=64, seed=123, pad_mode="reflect"):
        self.pairs = pairs
        self.patch_size = _to_3tuple(patch_size)
        self.ppv = patches_per_volume
        self.pad_mode = pad_mode
        random.seed(seed)

        self.meta = []
        for r, g in self.pairs:
            nii_r = nib.load(r); nii_g = nib.load(g)
            assert nii_r.shape == nii_g.shape, f"Shape mismatch: {r} vs {g}"
            self.meta.append({
                "raw": r, "gt": g,
                "shape": nii_r.shape,
                "affine": nii_r.affine
            })

    def __len__(self):
        return len(self.pairs) * self.ppv

    def _pad_to_min(self, vol, target):
        D, H, W = vol.shape
        td, th, tw = target
        pd, ph, pw = max(0, td-D), max(0, th-H), max(0, tw-W)
        if pd == ph == pw == 0:
            return vol
        pd0, pd1 = pd // 2, pd - pd // 2
        ph0, ph1 = ph // 2, ph - ph // 2
        pw0, pw1 = pw // 2, pw - pw // 2
        mode = "reflect" if self.pad_mode == "reflect" else "constant"
        return np.pad(vol, ((pd0,pd1),(ph0,ph1),(pw0,pw1)), mode=mode)

    def _rand_crop_coords(self, shape, patch):
        Dz, Hy, Wx = shape
        pd, ph, pw = patch
        z = random.randint(0, Dz - pd)
        y = random.randint(0, Hy - ph)
        x = random.randint(0, Wx - pw)
        return z, y, x

    def __getitem__(self, idx):
        vidx = idx // self.ppv
        info = self.meta[vidx]
        raw = nib.load(info["raw"]).get_fdata(dtype=np.float32)
        gt  = nib.load(info["gt"]).get_fdata(dtype=np.float32)
        raw = robust_norm(raw)
        gt  = robust_norm(gt)
        pdims = self.patch_size
        raw = self._pad_to_min(raw, pdims)
        gt  = self._pad_to_min(gt, pdims)
        z, y, x = self._rand_crop_coords(raw.shape, pdims)
        d, h, w = pdims
        x_patch = raw[z:z+d, y:y+h, x:x+w][None, ...]
        y_patch = gt [z:z+d, y:y+h, x:x+w][None, ...]
        return torch.from_numpy(x_patch), torch.from_numpy(y_patch)

# ----------------------------
# Metrics
# ----------------------------
def mae3d(pred, target):
    return torch.mean(torch.abs(pred - target))

def rmse3d(pred, target):
    return torch.sqrt(torch.mean((pred - target)**2))

def psnr3d(pred, target, data_range=1.0, eps=1e-8):
    mse = torch.mean((pred - target)**2)
    return 20.0 * torch.log10(torch.tensor(data_range, device=pred.device)) \
           - 10.0 * torch.log10(mse + eps)

def ncc3d(pred, target, eps=1e-8):
    p = pred - pred.mean()
    t = target - target.mean()
    num = torch.sum(p * t)
    den = torch.sqrt(torch.sum(p*p) * torch.sum(t*t) + eps)
    return num / (den + eps)

def _gaussian_kernel_3d(size=11, sigma=1.5, device="cpu"):
    ax = torch.arange(size, device=device) - size // 2
    k1 = torch.exp(-(ax**2) / (2 * sigma * sigma))
    k1 = (k1 / k1.sum())
    return (k1.view(1,1,size,1,1)
            * k1.view(1,1,1,size,1)
            * k1.view(1,1,1,1,size))

def ssim3d(pred, target, data_range=1.0,
           window_size=11, sigma=1.5, K1=0.01, K2=0.03):
    pred32, target32 = pred.float(), target.float()
    device = pred32.device
    ch = pred32.size(1)
    window = _gaussian_kernel_3d(window_size, sigma, device=device) \
        .expand(ch, 1, window_size, window_size, window_size)
    mu1 = F.conv3d(pred32, window, padding=window_size//2, groups=ch)
    mu2 = F.conv3d(target32, window, padding=window_size//2, groups=ch)
    mu1_sq, mu2_sq, mu1_mu2 = mu1*mu1, mu2*mu2, mu1*mu2
    sigma1_sq = F.conv3d(pred32*pred32, window,
                         padding=window_size//2, groups=ch) - mu1_sq
    sigma2_sq = F.conv3d(target32*target32, window,
                         padding=window_size//2, groups=ch) - mu2_sq
    sigma12   = F.conv3d(pred32*target32, window,
                         padding=window_size//2, groups=ch) - mu1_mu2
    C1 = (K1 * data_range)**2
    C2 = (K2 * data_range)**2
    ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-12)
    return ssim_map.mean()

def sobel3d(x):
    x32 = x.float()
    k = torch.tensor([[-1,0,1]], dtype=x32.dtype, device=x32.device)
    gx = F.conv3d(x32, k.view(1,1,1,1,3), padding=(0,0,1))
    gy = F.conv3d(x32, k.view(1,1,1,3,1), padding=(0,1,0))
    gz = F.conv3d(x32, k.view(1,1,3,1,1), padding=(1,0,0))
    return torch.sqrt(gx*gx + gy*gy + gz*gz + 1e-8)

def edge_ssim3d(pred, target):
    return ssim3d(sobel3d(pred), sobel3d(target))

class L1_SSIM_Loss(nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()
    def forward(self, pred, target):
        l1 = self.l1(pred, target)
        ssim = ssim3d(pred, target)
        return self.alpha * l1 + (1.0 - self.alpha) * (1.0 - ssim)

# ----------------------------
# Inference tiling helpers
# ----------------------------
def gaussian_weight(win):
    def g1(n):
        x = np.linspace(-1, 1, n)
        return np.exp(-4 * (x**2))
    wz, wy, wx = (g1(win[0]), g1(win[1]), g1(win[2]))
    W = np.outer(wz, wy).reshape(win[0], win[1], 1) * wx.reshape(1,1,win[2])
    return (W / W.max()).astype(np.float32)

def pad_to_multiple(vol, mult=(4,4,4), min_size=(96,96,96), mode="reflect"):
    D, H, W = vol.shape
    md, mh, mw = mult
    td = max(min_size[0], int(np.ceil(D / md) * md))
    th = max(min_size[1], int(np.ceil(H / mh) * mh))
    tw = max(min_size[2], int(np.ceil(W / mw) * mw))

    pd, ph, pw = max(0, td-D), max(0, th-H), max(0, tw-W)
    if pd == ph == pw == 0:
        return vol, (0,0,0), (D,H,W)

    pd0 = pd // 2; pd1 = pd - pd0
    ph0 = ph // 2; ph1 = ph - ph0
    pw0 = pw // 2; pw1 = pw - pw0
    out = np.pad(vol, ((pd0,pd1),(ph0,ph1),(pw0,pw1)), mode=mode)
    return out, (pd0,ph0,pw0), (D,H,W)

def unpad(vol, pads, orig_shape):
    pd0, ph0, pw0 = pads
    D, H, W = orig_shape
    return vol[pd0:pd0+D, ph0:ph0+H, pw0:pw0+W]

@torch.no_grad()
def infer_volume(model, vol, roi=(96,96,96),
                 overlap=0.5, device="cuda", use_amp=True):
    model.eval()
    D, H, W = vol.shape
    rd, rh, rw = roi
    sd = max(1, int(rd * (1 - overlap)))
    sh = max(1, int(rh * (1 - overlap)))
    sw = max(1, int(rw * (1 - overlap)))

    out = np.zeros((D,H,W), dtype=np.float32)
    wgt = np.zeros((D,H,W), dtype=np.float32)
    gw = gaussian_weight(roi)

    for z in range(0, max(1, D-rd+1), sd):
        for y in range(0, max(1, H-rh+1), sh):
            for x in range(0, max(1, W-rw+1), sw):
                zz, yy, xx = z, y, x
                z2, y2, x2 = zz+rd, yy+rh, xx+rw
                if z2 > D:
                    zz, z2 = D-rd, D
                if y2 > H:
                    yy, y2 = H-rh, H
                if x2 > W:
                    xx, x2 = W-rw, W
                patch = vol[zz:z2, yy:y2, xx:x2][None, None, ...]
                t = torch.from_numpy(patch.astype(np.float32)).to(device)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    logits = model(t)
                    pred = torch.sigmoid(logits)  # apply sigmoid at inference
                pred = torch.clamp(pred, 0.0, 1.0).float().cpu().numpy()[0,0]
                ww = gw[:pred.shape[0], :pred.shape[1], :pred.shape[2]]
                out[zz:z2, yy:y2, xx:x2] += pred * ww
                wgt[zz:z2, yy:y2, xx:x2] += ww
    out = out / np.maximum(wgt, 1e-6)
    return np.clip(out, 0.0, 1.0)

# ----------------------------
# Pairs loader
# ----------------------------
def load_pairs(pairs_path, sheet=None):
    pairs_path = _expand(pairs_path)
    ext = os.path.splitext(pairs_path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(pairs_path, sheet_name=sheet)
    elif ext == ".csv":
        df = pd.read_csv(pairs_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .xlsx, .xls, or .csv")

    df.columns = [c.strip().lower() for c in df.columns]
    required = {"raw", "gt"}
    if not required.issubset(df.columns):
        raise ValueError(f"Input must contain columns {required}. Found: {list(df.columns)}")

    df = df.dropna(subset=["raw","gt"]).copy()
    df["raw"] = df["raw"].astype(str).map(_expand)
    df["gt"]  = df["gt"].astype(str).map(_expand)

    kept, missing = [], []
    for r, g in zip(df["raw"], df["gt"]):
        ok = True
        if not os.path.exists(r):
            missing.append(("raw", r)); ok = False
        if not os.path.exists(g):
            missing.append(("gt", g)); ok = False
        if ok:
            kept.append((r,g))
    if missing:
        print("[WARN] Skipping rows with missing files:")
        for kind, p in missing:
            print(f"  - {kind} not found: {p}")
    if not kept:
        raise ValueError("No valid pairs after checking file existence.")
    return kept

# ----------------------------
# Train / Infer
# ----------------------------
def train(args):
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    pairs = load_pairs(args.pairs, sheet=args.sheet)
    assert len(pairs) >= 2, "Need at least 2 pairs (train/val)."

    random.shuffle(pairs)
    n_val = max(1, int(0.2 * len(pairs)))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    train_ds = PairedPatchDataset(
        train_pairs, patch_size=(args.patch,args.patch,args.patch),
        patches_per_volume=args.ppv, seed=123
    )
    val_ds   = PairedPatchDataset(
        val_pairs, patch_size=(args.patch,args.patch,args.patch),
        patches_per_volume=max(8, args.ppv//4), seed=456
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )
    val_loader   = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers
    )

    model = TinyUNETR3D(
        in_ch=1, base_ch=args.base_ch,
        window_size=(args.win,args.win,args.win)
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = L1_SSIM_Loss(alpha=args.ssim_alpha)
    use_amp = (device == "cuda") and (not args.no_amp)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=2,
        min_lr=1e-6, verbose=False
    )

    best = float('inf')
    no_improve = 0
    history = {k: [] for k in
               ["epoch","train_loss","val_loss","MAE","RMSE",
                "PSNR","SSIM","EdgeSSIM","NCC"]}

    for epoch in range(1, args.epochs+1):
        model.train()
        tl = 0.0
        for x, y in tqdm(train_loader, desc=f"epoch {epoch} train", ncols=100):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(x)
                p = torch.sigmoid(logits)  # sigmoid here (not in model)
                if not torch.isfinite(p).all():
                    print("[WARN] NaN/Inf in output; halving LR & disabling AMP.")
                    for g in opt.param_groups:
                        g['lr'] = max(g['lr'] * 0.5, 1e-6)
                    use_amp = False
                    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
                    p = torch.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0)
                p = torch.clamp(p, 0.0, 1.0)
                loss = loss_fn(p, y)
                if not torch.isfinite(loss):
                    print("[WARN] NaN/Inf loss; skip batch.")
                    continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(opt)
            scaler.update()
            tl += loss.item() * x.size(0)
        tl /= len(train_loader.dataset)

        # validation
        model.eval()
        vl = 0.0
        v_mae = v_rmse = v_psnr = v_ssim = v_essim = v_ncc = 0.0
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
            for x, y in tqdm(val_loader, desc=f"epoch {epoch} val  ", ncols=100):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                p = torch.sigmoid(logits)
                if not torch.isfinite(p).all():
                    p = torch.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0)
                p = torch.clamp(p, 0.0, 1.0)
                l = loss_fn(p, y)
                if not torch.isfinite(l):
                    continue
                vl += l.item() * x.size(0)
                v_mae  += mae3d(p, y).item() * x.size(0)
                v_rmse += rmse3d(p, y).item() * x.size(0)
                v_psnr += psnr3d(p, y).item() * x.size(0)
                v_ssim += ssim3d(p, y).item() * x.size(0)
                v_essim+= edge_ssim3d(p, y).item() * x.size(0)
                v_ncc  += ncc3d(p, y).item() * x.size(0)

        n = len(val_loader.dataset)
        if n > 0:
            vl    /= n
            v_mae /= n
            v_rmse/= n
            v_psnr/= n
            v_ssim/= n
            v_essim/= n
            v_ncc /= n
        else:
            vl = v_mae = v_rmse = v_psnr = v_ssim = v_essim = v_ncc = float('nan')

        print(
            f"Epoch {epoch:03d}  train {tl:.4f}  val {vl:.4f}  |  "
            f"MAE {v_mae:.4f}  RMSE {v_rmse:.4f}  PSNR {v_psnr:.2f}  "
            f"SSIM {v_ssim:.4f}  EdgeSSIM {v_essim:.4f}  NCC {v_ncc:.4f}"
        )

        if np.isfinite(vl):
            scheduler.step(vl)

        improvement = (np.isfinite(vl) and vl < best - 1e-6)
        ckpt_path = ""
        if improvement:
            best = vl
            no_improve = 0
            Path(args.out_dir).mkdir(parents=True, exist_ok=True)
            ckpt_path = os.path.join(args.out_dir, "best_3d_deconv.pt")
            torch.save(model.state_dict(), ckpt_path)
            with open(os.path.join(args.out_dir, "train_log.json"), "w") as f:
                json.dump({
                    "best_val": float(best),
                    "epoch": epoch,
                    "mae": float(v_mae),
                    "rmse": float(v_rmse),
                    "psnr": float(v_psnr),
                    "ssim": float(v_ssim),
                    "edge_ssim": float(v_essim),
                    "ncc": float(v_ncc),
                }, f)
        else:
            no_improve += 1

        row = {
            "epoch": epoch,
            "train_loss": float(tl),
            "val_loss": float(vl),
            "MAE": float(v_mae),
            "RMSE": float(v_rmse),
            "PSNR": float(v_psnr),
            "SSIM": float(v_ssim),
            "EdgeSSIM": float(v_essim),
            "NCC": float(v_ncc),
            "lr": float(opt.param_groups[0]["lr"]),
            "params": int(count_params(model)),
            "is_best": 1 if improvement else 0,
            "ckpt_path": ckpt_path
        }
        _append_row_csv(args.train_csv, row, header=list(row.keys()))
        if improvement and args.best_csv:
            _append_row_csv(args.best_csv, row, header=list(row.keys()))

        history["epoch"].append(epoch)
        for k in ["train_loss","val_loss","MAE","RMSE","PSNR",
                  "SSIM","EdgeSSIM","NCC"]:
            history[k].append(row[k])

        if (not improvement) and no_improve >= args.early_stop:
            print("Early stopping triggered.")
            break

    if args.plot_curves:
        _plot_curves(args.out_dir, history)
        print(f"Saved curves to {args.out_dir}")

def _print_metrics_header_once():
    hdr = ["RUN","IN_NII","GT_NII","BASE_CH","WIN","ROI","OVERLAP",
           "MAE","RMSE","PSNR","SSIM","EdgeSSIM","NCC","PARAMS","INFER_SEC","CKPT"]
    print("| " + " | ".join(hdr) + " |")

def _fmt(x, nd=6):
    if x is None:
        return "—"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)

@torch.no_grad()
def infer(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model = TinyUNETR3D(
        in_ch=1, base_ch=args.base_ch,
        window_size=(args.win,args.win,args.win)
    ).to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # count params now
    n_params = count_params(model)

    in_path = Path(args.in_nii)
    nii = nib.load(str(in_path))
    vol = robust_norm(nii.get_fdata(dtype=np.float32))

    if args.pad_win:
        vol_pad, pads, orig_shape = pad_to_multiple(
            vol, mult=(args.win,args.win,args.win),
            min_size=(args.roi,args.roi,args.roi), mode="reflect"
        )
    else:
        vol_pad, pads, orig_shape = vol, (0,0,0), vol.shape

    use_amp = (device == "cuda") and (not args.no_amp)
    t0 = time.time()
    pred_pad = infer_volume(
        model, vol_pad, roi=(args.roi,args.roi,args.roi),
        overlap=args.overlap, device=device, use_amp=use_amp
    )
    infer_time_sec = time.time() - t0
    pred = unpad(pred_pad, pads, orig_shape) if args.pad_win else pred_pad

    out_nii = nib.Nifti1Image(pred.astype(np.float32),
                              affine=nii.affine, header=nii.header)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out_dir) / (in_path.stem.replace(".nii","") + "_pred.nii.gz")
    nib.save(out_nii, str(out_path))
    print(f"Saved: {out_path}")

    # default metric fields
    mae = rmse = psnr = ssim = essim = ncc = None

    # compute metrics only if GT provided
    if args.gt_nii is not None and os.path.exists(args.gt_nii):
        gt_vol = robust_norm(nib.load(args.gt_nii).get_fdata(dtype=np.float32))
        D = min(pred.shape[0], gt_vol.shape[0])
        H = min(pred.shape[1], gt_vol.shape[1])
        W = min(pred.shape[2], gt_vol.shape[2])
        pred_t = torch.from_numpy(pred[:D,:H,:W]).unsqueeze(0).unsqueeze(0).to(device).float()
        gt_t   = torch.from_numpy(gt_vol[:D,:H,:W]).unsqueeze(0).unsqueeze(0).to(device).float()
        mae  = float(mae3d(pred_t, gt_t).item())
        rmse = float(rmse3d(pred_t, gt_t).item())
        psnr = float(psnr3d(pred_t, gt_t).item())
        ssim = float(ssim3d(pred_t, gt_t).item())
        essim= float(edge_ssim3d(pred_t, gt_t).item())
        ncc  = float(ncc3d(pred_t, gt_t).item())
        print(
            f"[INFER METRICS] MAE {mae:.4f}  RMSE {rmse:.4f}  "
            f"PSNR {psnr:.2f}  SSIM {ssim:.4f}  EdgeSSIM {essim:.4f}  NCC {ncc:.4f}"
        )
    else:
        print("[INFER] No --gt_nii provided; metrics skipped (params still shown below).")

    # pretty one-line table print (always)
    _print_metrics_header_once()
    row = [
        (args.save_prefix or "run"),
        str(in_path),
        (str(args.gt_nii) if args.gt_nii else "—"),
        args.base_ch, args.win, args.roi, args.overlap,
        _fmt(mae), _fmt(rmse), _fmt(psnr, nd=3),
        _fmt(ssim), _fmt(essim), _fmt(ncc),
        int(n_params), _fmt(infer_time_sec, nd=3), args.weights
    ]
    print("| " + " | ".join(map(str, row)) + " |")

    # optional CSV write
    if args.gt_nii and os.path.exists(args.gt_nii) and args.metrics_csv:
        header = ["run","in_nii","gt_nii","base_ch","win","roi","overlap",
                  "MAE","RMSE","PSNR","SSIM","EdgeSSIM","NCC",
                  "params","infer_sec","ckpt_used"]
        Path(os.path.dirname(args.metrics_csv)).mkdir(parents=True, exist_ok=True)
        write_header = not os.path.exists(args.metrics_csv)
        with open(args.metrics_csv, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(header)
            w.writerow([
                (args.save_prefix or "run"),
                str(in_path), str(args.gt_nii),
                args.base_ch, args.win, args.roi, args.overlap,
                _fmt(mae), _fmt(rmse), _fmt(psnr, nd=3),
                _fmt(ssim), _fmt(essim), _fmt(ncc),
                int(n_params), _fmt(infer_time_sec, nd=3), args.weights
            ])

# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--pairs", required=True, help="Excel/CSV with columns raw,gt")
    tr.add_argument("--sheet", default=None)
    tr.add_argument("--out_dir", default="runs")
    tr.add_argument("--epochs", type=int, default=100)
    tr.add_argument("--batch", type=int, default=2)
    tr.add_argument("--patch", type=int, default=64)
    tr.add_argument("--ppv", type=int, default=64)
    tr.add_argument("--workers", type=int, default=4)
    tr.add_argument("--base_ch", type=int, default=24)
    tr.add_argument("--win", type=int, default=4)
    tr.add_argument("--lr", type=float, default=5e-5)  # safer default
    tr.add_argument("--ssim_alpha", type=float, default=0.85)
    tr.add_argument("--early_stop", type=int, default=6)
    tr.add_argument("--no_amp", action="store_true")
    tr.add_argument("--cpu", action="store_true")

    tr.add_argument("--train_csv", default=None,
                    help="Per-epoch CSV (default <out_dir>/train_history.csv)")
    tr.add_argument("--plot_curves", action="store_true")
    tr.add_argument("--best_csv", default=None,
                    help="CSV recording only best epochs")

    tr.set_defaults(func=train)

    inf = sub.add_parser("infer")
    inf.add_argument("--weights", required=True)
    inf.add_argument("--in_nii", required=True)
    inf.add_argument("--gt_nii", default=None)
    inf.add_argument("--out_dir", default="preds")
    inf.add_argument("--roi", type=int, default=96)
    inf.add_argument("--overlap", type=float, default=0.5)
    inf.add_argument("--base_ch", type=int, default=24)
    inf.add_argument("--win", type=int, default=4)
    inf.add_argument("--pad_win", action="store_true")
    inf.add_argument("--metrics_csv", default=None)
    inf.add_argument("--save_prefix", default=None)
    inf.add_argument("--no_amp", action="store_true")
    inf.add_argument("--cpu", action="store_true")
    inf.set_defaults(func=infer)

    args = ap.parse_args()

    if args.cmd == "train":
        if not args.train_csv or args.train_csv.strip() == "":
            args.train_csv = str(Path(args.out_dir) / "train_history.csv")
        if not args.best_csv or args.best_csv.strip() == "":
            args.best_csv  = str(Path(args.out_dir) / "best_history.csv")

    args.func(args)

if __name__ == "__main__":
    main()

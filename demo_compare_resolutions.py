#!/usr/bin/env python3
import argparse, sys, numpy as np, torch
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

# Regional architecture (nn.Module, not Lightning)
from climax.regional_forecast.arch import RegionalClimaX

# ---- Variables list (order must match training!) ----
DEFAULT_VARS = [
    "land_sea_mask","orography","lattitude",
    "2m_temperature","10m_u_component_of_wind","10m_v_component_of_wind",
    "geopotential_50","geopotential_250","geopotential_500",
    "geopotential_600","geopotential_700","geopotential_850","geopotential_925",
    "u_component_of_wind_50","u_component_of_wind_250","u_component_of_wind_500",
    "u_component_of_wind_600","u_component_of_wind_700","u_component_of_wind_850","u_component_of_wind_925",
    "v_component_of_wind_50","v_component_of_wind_250","v_component_of_wind_500",
    "v_component_of_wind_600","v_component_of_wind_700","v_component_of_wind_850","v_component_of_wind_925",
    "temperature_50","temperature_250","temperature_500","temperature_600",
    "temperature_700","temperature_850","temperature_925",
    "relative_humidity_50","relative_humidity_250","relative_humidity_500","relative_humidity_600",
    "relative_humidity_700","relative_humidity_850","relative_humidity_925",
    "specific_humidity_50","specific_humidity_250","specific_humidity_500","specific_humidity_600",
    "specific_humidity_700","specific_humidity_850","specific_humidity_925",
]
DEFAULT_OUT_VARS = ["geopotential_500","temperature_850","2m_temperature"]

def parse_args():
    p = argparse.ArgumentParser(description="ClimaX predict-only: compare resolutions side-by-side")
    p.add_argument("--ckpt_5625", required=True, help="Path to 5.625° checkpoint")
    p.add_argument("--ckpt_1406", required=True, help="Path to 1.40625° checkpoint")
    p.add_argument("--device", choices=["mps","cpu"], default="mps")
    p.add_argument("--out_vars", nargs="*", default=DEFAULT_OUT_VARS)
    p.add_argument("--batch", type=int, default=1)
    # Arch hyperparams (should match ckpts; defaults work for official ones)
    p.add_argument("--embed_dim", type=int, default=1024)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--decoder_depth", type=int, default=2)
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--drop_path", type=float, default=0.1)
    p.add_argument("--drop_rate", type=float, default=0.1)
    return p.parse_args()

def geo_for(res: str) -> Tuple[Tuple[int,int], int]:
    return ((32,64), 2) if res == "5.625" else ((128,256), 4)

def build_region_info(H: int, W: int, patch: int) -> Dict[str, np.ndarray]:
    lat_vec = np.linspace(-90.0, 90.0, num=H, dtype=np.float32)
    lon_vec = np.linspace(0.0, 360.0, num=W, endpoint=False, dtype=np.float32)
    hp, wp = H // patch, W // patch
    L = hp * wp
    patch_ids = np.arange(L, dtype=np.int64)
    return {
        # full-frame ranges
        "min_h": 0, "max_h": H, "min_w": 0, "max_w": W,
        # per-pixel indices
        "h_inds": np.arange(H, dtype=np.int64),
        "w_inds": np.arange(W, dtype=np.int64),
        # patch grid
        "patch": patch, "hp": hp, "wp": wp, "patch_ids": patch_ids,
        # metadata
        "lat": lat_vec, "lon": lon_vec,
        "north": 90.0, "south": -90.0, "west": 0.0, "east": 360.0,
        "name": "full",
    }

def strip_prefixes(sd: dict) -> dict:
    for p in ("model.","net.","module.","climax."):
        if any(k.startswith(p) for k in sd):
            return {k[len(p):]: v for k,v in sd.items()}
    return sd

def load_checkpoint(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    return strip_prefixes(sd)

class CaptureMetric:
    """Metric that captures predictions and returns zero loss."""
    def __init__(self):
        self.last_pred = None
    def __call__(self, preds: torch.Tensor, y_true: torch.Tensor, out_vars, lat_vec: torch.Tensor):
        self.last_pred = preds
        return torch.tensor(0.0, device=preds.device, dtype=preds.dtype), {}

def predict_one(ckpt_path: str, res: str, device: torch.device, out_vars: List[str],
                embed_dim: int, depth: int, decoder_depth: int, num_heads: int,
                mlp_ratio: float, drop_path: float, drop_rate: float, batch: int):
    (H, W), patch = geo_for(res)
    model = RegionalClimaX(DEFAULT_VARS, (H, W), patch,
                           embed_dim, depth, decoder_depth,
                           num_heads, mlp_ratio, drop_path, drop_rate)
    sd = load_checkpoint(ckpt_path)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[{res}] loaded | missing: {len(missing)} unexpected: {len(unexpected)}")
    x   = torch.randn(batch, len(DEFAULT_VARS), H, W, dtype=torch.float32, device=device)
    y   = torch.zeros(batch, len(out_vars), H, W, dtype=torch.float32, device=device)
    lat = torch.linspace(-90.0, 90.0, steps=H, dtype=torch.float32, device=device)
    lead_times = torch.tensor([1], dtype=torch.float32, device=device)
    region_info = build_region_info(H, W, patch)
    metric = CaptureMetric()
    model.to(device).eval()
    with torch.no_grad():
        _ = model(x, y, lead_times, DEFAULT_VARS, out_vars, [metric], lat, region_info)
    preds = metric.last_pred  # (B, V, H, W)
    if preds is None:
        raise RuntimeError("Metric didn’t receive predictions; forward signature mismatch.")
    return preds.detach().cpu().numpy()[0], (H, W)  # (V,H,W), (H,W)

def plot_side_by_side(var_name: str, arr_lo: np.ndarray, arr_hi: np.ndarray,
                      shape_lo: Tuple[int,int], shape_hi: Tuple[int,int], out_path: str):
    plt.figure(figsize=(10,4))
    plt.suptitle(f"{var_name} — ClimaX predictions (demo)")

    plt.subplot(1,2,1)
    plt.title("5.625° (32×64)")
    plt.imshow(arr_lo, origin="lower")  # no explicit colormap/colors per your environment rules
    plt.colorbar(shrink=0.8)

    plt.subplot(1,2,2)
    plt.title("1.40625° (128×256)")
    plt.imshow(arr_hi, origin="lower")
    plt.colorbar(shrink=0.8)

    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(out_path, dpi=150)
    plt.close()

def main():
    args = parse_args()
    device = torch.device("mps" if (args.device=="mps" and torch.backends.mps.is_available()) else "cpu")

    # Predict on both resolutions
    preds_lo, shape_lo = predict_one(
        args.ckpt_5625, "5.625", device, args.out_vars,
        args.embed_dim, args.depth, args.decoder_depth, args.num_heads,
        args.mlp_ratio, args.drop_path, args.drop_rate, args.batch
    )
    preds_hi, shape_hi = predict_one(
        args.ckpt_1406, "1.40625", device, args.out_vars,
        args.embed_dim, args.depth, args.decoder_depth, args.num_heads,
        args.mlp_ratio, args.drop_path, args.drop_rate, args.batch
    )

    # Save raw tensors (optional)
    np.save("preds_5p625.npy", preds_lo)  # (V, 32, 64)
    np.save("preds_1p406.npy", preds_hi)  # (V, 128, 256)

    # Side-by-side plots per variable
    for i, v in enumerate(args.out_vars):
        out_png = f"compare_{v.replace('/', '_')}.png"
        plot_side_by_side(v, preds_lo[i], preds_hi[i], shape_lo, shape_hi, out_png)
        print(f"saved {out_png}")

    print("Done.")

if __name__ == "__main__":
    main()

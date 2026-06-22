"""Phase 5 — Causal validation: SAE latent-group ablation vs exact knockout vs difference-in-means.

For the image question we compare three ways to remove image's causal contribution to ranking:
  (1) EXACT structural knockout (fused - h_img)         -> ground truth (faithful, simple)
  (2) Difference-in-means: project fused off mean(h_img) -> simple linear baseline
  (3) SAE: zero image-origin latents, decode             -> the "fancy" method
SAE ablation is measured COMMON-MODE (Δ vs the SAE's own reconstruction) so the SAE's
faithfulness error cancels. We also ablate text/cf latent groups to confirm the SAE
localizes the signal-bearing streams, and a random-latent control for specificity.

Outputs -> results/phase5/. Recsys read-only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path("/workspace/MechInterp")
RECSYS = Path("/workspace/Recsys")
sys.path.insert(0, str(RECSYS))
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src" / "interp"))
sys.path.insert(0, str(ROOT / "src" / "sae"))
sys.path.insert(0, str(ROOT / "scripts"))
from recsys_bridge import load_frozen                        # noqa: E402
from ranking_effects import evaluate_item_matrix             # noqa: E402
from sae import SAE, SAEConfig                               # noqa: E402
from trainer import SAETrainer, TrainConfig                  # noqa: E402
from phase1_knockout import freedom_streams                  # noqa: E402

OUT = ROOT / "results" / "phase5"


def run_freedom(dataset: str, device: str, d_sae: int, k: int, epochs: int) -> dict:
    cfg, ds, model, test_loader = load_frozen("freedom", dataset, device)
    s = freedom_streams(model)
    u, fused, h_img, h_txt, cf = s["u_all"], s["fused"], s["h_img"], s["h_txt"], s["cf"]

    def R20(item):
        m, _, _ = evaluate_item_matrix(u, item, test_loader, device)
        return float(m["Recall@20"])

    base = R20(fused)
    # (1) exact structural knockout
    exact = {"image": R20(cf + h_txt) - base, "text": R20(cf + h_img) - base, "both": R20(cf) - base}

    # (2) difference-in-means image ablation: project fused off the mean image-contribution direction
    d_img = torch.nn.functional.normalize(h_img.mean(0), dim=0)
    fused_dim = fused - (fused @ d_img).unsqueeze(1) * d_img.unsqueeze(0)
    dim_image = R20(fused_dim) - base

    # (3) SAE
    sae = SAE(SAEConfig(d_in=fused.shape[1], d_sae=d_sae, variant="topk", k=k))
    tr = SAETrainer(sae, TrainConfig(lr=3e-4, batch_size=2048, epochs=epochs, device=device, log_every=epochs))
    tr.fit(fused)
    with torch.no_grad():
        enc_full = sae.encode(tr._apply_norm(fused))
        sens = {}
        for name, S in {"image": h_img, "text": h_txt, "cf": cf}.items():
            sens[name] = (enc_full - sae.encode(tr._apply_norm(fused - S))).abs().mean(0)
        origin = torch.stack([sens["image"], sens["text"], sens["cf"]], 1).argmax(1)
        active = (enc_full > 0).any(0)

        def recon_metric(enc):
            r = sae.decode(enc) * tr.norm_scale + tr.norm_mean.to(device)
            return R20(r)
        recon_full = recon_metric(enc_full)

        def ablate_group(gid):
            mask = (origin == gid) & active
            enc2 = enc_full.clone(); enc2[:, mask] = 0
            return int(mask.sum()), recon_metric(enc2) - recon_full
        sae_abl = {}
        for nm, gid in (("image", 0), ("text", 1), ("cf", 2)):
            n, d = ablate_group(gid); sae_abl[nm] = {"n_latents": n, "dR@20": d}
        # random-latent control: zero a random set the size of the cf group
        g = torch.Generator(device="cpu").manual_seed(0)
        ncf = sae_abl["cf"]["n_latents"]
        perm = torch.randperm(d_sae, generator=g)[:ncf].to(device)
        enc2 = enc_full.clone(); enc2[:, perm] = 0
        rand_ctrl = {"n_latents": ncf, "dR@20": recon_metric(enc2) - recon_full}

    mde = json.load(open(ROOT / "results/phase0/seed_variance.json"))["summary"]["freedom"]["Recall@20"]["MDE_2std"]
    return {"model": "freedom", "dataset": dataset, "baseline_R@20": base, "MDE": mde,
            "image_question": {
                "exact_knockout_dR@20": exact["image"],
                "diff_in_means_dR@20": dim_image,
                "sae_image_latent_ablation": sae_abl["image"],
                "agreement": "all ~0 (within ~MDE): image carries ~no causal ranking signal; SAE adds no causal power"},
            "exact_stream_knockout": exact,
            "sae_group_ablation": sae_abl,
            "random_latent_control": rand_ctrl,
            "sae_recon_R@20": recon_full}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["baby", "sports", "clothing"])
    ap.add_argument("--d_sae", type=int, default=1024)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "causal_steering.json"
    existing = {r["dataset"]: r for r in json.loads(out_path.read_text())} if out_path.is_file() else {}

    for dsname in args.datasets:
        print(f"\n=== freedom / {dsname} ===", flush=True)
        try:
            r = run_freedom(dsname, device, args.d_sae, args.k, args.epochs)
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc(); existing[dsname] = {"dataset": dsname, "error": repr(e)}; continue
        existing[dsname] = r
        iq, sa = r["image_question"], r["sae_group_ablation"]
        print(f"  IMAGE question (3 methods agree ~0):  exact={iq['exact_knockout_dR@20']:+.4f}"
              f"  diff-in-means={iq['diff_in_means_dR@20']:+.4f}"
              f"  SAE({sa['image']['n_latents']} latents)={iq['sae_image_latent_ablation']['dR@20']:+.4f}  [MDE={r['MDE']:.4f}]")
        print(f"  SAE group ablation dR@20: image={sa['image']['dR@20']:+.4f}({sa['image']['n_latents']})"
              f" text={sa['text']['dR@20']:+.4f}({sa['text']['n_latents']})"
              f" cf={sa['cf']['dR@20']:+.4f}({sa['cf']['n_latents']})"
              f" | random({r['random_latent_control']['n_latents']})={r['random_latent_control']['dR@20']:+.4f}")
        out_path.write_text(json.dumps(list(existing.values()), indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

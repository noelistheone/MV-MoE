"""Phase 4 — SAE decomposition of the fused stream + modality-origin attribution.

FREEDOM's fused item embedding decomposes EXACTLY as fused = cf + h_img + h_txt, so a
latent's modality origin is measured causally: how much its activation changes when a
stream is removed from the input. We also gate on SAE faithfulness (does decode(encode)
preserve R@20?) and report FVE/L0/dead.

Prediction (consistency with Phases 1-3): since ‖h_img‖ is tiny, the SAE on FREEDOM's
fused stream should contain almost no image-origin structure.

Outputs -> results/phase4/. Recsys read-only.
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

OUT = ROOT / "results" / "phase4"


def run_freedom(dataset: str, device: str, d_sae: int, k: int, epochs: int) -> dict:
    cfg, ds, model, test_loader = load_frozen("freedom", dataset, device)
    s = freedom_streams(model)
    u_all, fused, h_img, h_txt, cf = s["u_all"], s["fused"], s["h_img"], s["h_txt"], s["cf"]

    base_metrics, _, _ = evaluate_item_matrix(u_all, fused, test_loader, device)

    sae = SAE(SAEConfig(d_in=fused.shape[1], d_sae=d_sae, variant="topk", k=k))
    tr = SAETrainer(sae, TrainConfig(lr=3e-4, batch_size=2048, epochs=epochs,
                                     device=device, log_every=max(1, epochs)))
    hist = tr.fit(fused)
    fve, l0, dead = hist["fve"][-1], hist["l0"][-1], hist["dead_frac"][-1]

    with torch.no_grad():
        fn = tr._apply_norm(fused)
        recon_n, enc_full = sae(fn)
        recon = recon_n * tr.norm_scale + tr.norm_mean.to(device)
        faith_metrics, _, _ = evaluate_item_matrix(u_all, recon, test_loader, device)

        # modality-origin attribution via exact stream-removal sensitivity
        sens = {}
        for name, S in {"image": h_img, "text": h_txt, "cf": cf}.items():
            enc_minus = sae.encode(tr._apply_norm(fused - S))
            sens[name] = (enc_full - enc_minus).abs().mean(0)        # [d_sae]
        sens_mat = torch.stack([sens["image"], sens["text"], sens["cf"]], dim=1)
        active = (enc_full > 0).any(0)
        origin = sens_mat.argmax(dim=1)
        counts = {"image": int(((origin == 0) & active).sum()),
                  "text": int(((origin == 1) & active).sum()),
                  "cf": int(((origin == 2) & active).sum()),
                  "dead": int((~active).sum())}
        sens_mass = {nm: float(v.sum()) for nm, v in sens.items()}
        total = sum(sens_mass.values()) + 1e-12
        sens_frac = {nm: v / total for nm, v in sens_mass.items()}

    mde = None
    sv = ROOT / "results" / "phase0" / "seed_variance.json"
    if sv.is_file():
        mde = json.loads(sv.read_text()).get("summary", {}).get("freedom", {}).get("Recall@20", {}).get("MDE_2std")
    dR = float(faith_metrics["Recall@20"]) - float(base_metrics["Recall@20"])

    return {"model": "freedom", "dataset": dataset, "d_sae": d_sae, "k": k,
            "sae_quality": {"FVE": fve, "L0": l0, "dead_frac": dead},
            "faithfulness": {"baseline_R@20": float(base_metrics["Recall@20"]),
                             "sae_recon_R@20": float(faith_metrics["Recall@20"]),
                             "dR@20": dR, "MDE": mde,
                             "passes_gate": (abs(dR) <= (mde or 1e9))},
            "modality_origin_latent_counts": counts,
            "modality_origin_sensitivity_fraction": sens_frac}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["baby", "sports", "clothing"])
    ap.add_argument("--d_sae", type=int, default=512)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    out_path = OUT / "sae_attribution.json"
    existing = {}
    if out_path.is_file():
        for r in json.loads(out_path.read_text()):
            existing[f"{r['model']}/{r['dataset']}"] = r

    for dsname in args.datasets:
        print(f"\n=== freedom / {dsname} (SAE d_sae={args.d_sae} k={args.k}) ===", flush=True)
        try:
            r = run_freedom(dsname, device, args.d_sae, args.k, args.epochs)
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc()
            existing[f"freedom/{dsname}"] = {"model": "freedom", "dataset": dsname, "error": repr(e)}
            continue
        existing[f"freedom/{dsname}"] = r
        q, f, c = r["sae_quality"], r["faithfulness"], r["modality_origin_latent_counts"]
        sf = r["modality_origin_sensitivity_fraction"]
        print(f"  SAE: FVE={q['FVE']:.3f} L0={q['L0']:.1f} dead={q['dead_frac']:.2f}")
        print(f"  faithfulness: base R@20={f['baseline_R@20']:.4f} recon R@20={f['sae_recon_R@20']:.4f}"
              f" dR={f['dR@20']:+.4f} gate={'PASS' if f['passes_gate'] else 'FAIL'}")
        print(f"  latent origin counts: image={c['image']} text={c['text']} cf={c['cf']} dead={c['dead']}")
        print(f"  sensitivity fraction: image={sf['image']:.3f} text={sf['text']:.3f} cf={sf['cf']:.3f}")
        out_path.write_text(json.dumps(list(existing.values()), indent=2))

    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

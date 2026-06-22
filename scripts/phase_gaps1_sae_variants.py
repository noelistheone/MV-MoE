"""Phase 4 extensions (plan-specified, previously unrun):

(1) PREDICTION-AWARE SAE (port of the interaction-aware idea, arXiv:2511.18024):
    add a loss term that backprops through the frozen scorer so the sparse code
    preserves user-item affinity:  L_pred = E_{(u,i) in train} ( u·x̂_i − u·x_i )².
(2) CROSS-MODAL-MASKING SAE (adaptation of the split-dictionary fix, arXiv:2601.20028):
    FREEDOM's fusion is ADDITIVE (fused = cf + h_img + h_txt), not concatenated, so the
    faithful adaptation of "cross-modal masking" is: with some probability feed the SAE
    a stream-dropped input (fused − h_img or fused − h_txt) and require it to reconstruct
    the FULL fused embedding — encouraging latents that bind streams jointly.
(3) EXACT GRADIENT ATTRIBUTION cross-check for modality origin: the encoder pre-activation
    is linear, so stream S's contribution to latent j is exactly E_i |S_i · W_enc[:,j]| / scale.
    Cross-check against the stream-removal sensitivity used in phase4.

Per plan: run WITH and WITHOUT each variant; only claim benefit if the modality-origin
histogram / FVE / faithfulness actually changes. Outputs -> results/phase4/sae_variants.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
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
from phase1_knockout import freedom_streams                  # noqa: E402

OUT = ROOT / "results" / "phase4"


def fit_norm(x):
    mean = x.mean(0, keepdim=True)
    centered = x - mean
    scale = max((centered.pow(2).sum(-1).mean().sqrt() / (x.shape[1] ** 0.5)).item(), 1e-6)
    return mean, scale


def train_sae_variant(fused, streams, train_pairs, u_all, device,
                      d_sae=1024, k=32, epochs=400, variant="plain",
                      lambda_pred=1.0, mask_prob=0.5, seed=0):
    """variant: plain | pred (prediction-aware) | mask (cross-modal masking) | pred+mask."""
    torch.manual_seed(seed)
    mean, scale = fit_norm(fused)
    fn = (fused - mean) / scale
    sae = SAE(SAEConfig(d_in=fused.shape[1], d_sae=d_sae, variant="topk", k=k)).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=3e-4)
    n = fused.shape[0]
    bs = 2048
    h_img, h_txt = streams["h_img"], streams["h_txt"]
    pu, pi = train_pairs            # train (user, item) index tensors on device

    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for s0 in range(0, n - bs + 1, bs):
            idx = perm[s0:s0 + bs]
            x_tgt = fn[idx]
            x_in = x_tgt
            if "mask" in variant and torch.rand(1).item() < mask_prob:
                drop = h_img if torch.rand(1).item() < 0.5 else h_txt
                x_in = (fused[idx] - drop[idx] - mean) / scale     # stream-dropped input
            pre = (x_in - sae.b_dec) @ sae.W_enc + sae.b_enc
            acts = torch.relu(pre)
            topv, topi = acts.topk(k, dim=-1)
            acts = torch.zeros_like(acts).scatter_(-1, topi, topv)
            recon = acts @ sae.W_dec + sae.b_dec
            loss = (recon - x_tgt).pow(2).sum(-1).mean()
            sae._update_activity(acts)
            # AuxK on dead latents (same as SAE.loss)
            dead = sae.steps_since_active > sae.cfg.dead_steps_threshold
            if int(dead.sum()) > 0:
                k_aux = min(sae.cfg.aux_k, int(dead.sum()))
                masked = torch.relu(pre).masked_fill(~dead, 0.0)
                tv, ti = masked.topk(k_aux, dim=-1)
                aux_acts = torch.zeros_like(masked).scatter_(-1, ti, tv)
                loss = loss + sae.cfg.aux_coeff * (aux_acts @ sae.W_dec - (x_tgt - recon).detach()).pow(2).sum(-1).mean()
            if "pred" in variant:
                # sample a chunk of train pairs; preserve u·x̂ vs u·x in ORIGINAL space
                pidx = torch.randint(0, pu.numel(), (bs,), device=device)
                uu, ii = pu[pidx], pi[pidx]
                xi = fn[ii]
                pre2 = (xi - sae.b_dec) @ sae.W_enc + sae.b_enc
                a2 = torch.relu(pre2)
                tv2, ti2 = a2.topk(k, dim=-1)
                a2 = torch.zeros_like(a2).scatter_(-1, ti2, tv2)
                xhat = (a2 @ sae.W_dec + sae.b_dec) * scale + mean       # back to original space
                s_hat = (u_all[uu] * xhat).sum(-1)
                s_true = (u_all[uu] * fused[ii]).sum(-1)
                loss = loss + lambda_pred * (s_hat - s_true).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                sae.W_dec.div_(sae.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8))
    return sae, mean, scale


@torch.no_grad()
def analyze(sae, mean, scale, fused, streams, u_all, test_loader, device):
    fn = (fused - mean) / scale
    enc_full = sae.encode(fn)
    recon = sae.decode(enc_full) * scale + mean
    m_base, _, _ = evaluate_item_matrix(u_all, fused, test_loader, device)
    m_rec, _, _ = evaluate_item_matrix(u_all, recon, test_loader, device)
    # FVE
    resid = (recon - fused).pow(2).sum(-1).mean()
    tot = (fused - fused.mean(0, keepdim=True)).pow(2).sum(-1).mean()
    fve = float(1 - resid / tot)
    # origin via stream-removal sensitivity
    sens = {}
    for nm in ("h_img", "h_txt", "cf"):
        sens[nm] = (enc_full - sae.encode((fused - streams[nm] - mean) / scale)).abs().mean(0)
    smat = torch.stack([sens["h_img"], sens["h_txt"], sens["cf"]], 1)
    active = (enc_full > 0).any(0)
    origin_rm = smat.argmax(1)
    counts_rm = {nm: int(((origin_rm == i) & active).sum()) for i, nm in enumerate(("image", "text", "cf"))}
    # origin via EXACT linear-encoder gradient attribution: E_i |S_i · W_enc[:,j]| / scale
    grad = {}
    for nm in ("h_img", "h_txt", "cf"):
        grad[nm] = (streams[nm] @ sae.W_enc).abs().mean(0) / scale
    gmat = torch.stack([grad["h_img"], grad["h_txt"], grad["cf"]], 1)
    origin_gr = gmat.argmax(1)
    counts_gr = {nm: int(((origin_gr == i) & active).sum()) for i, nm in enumerate(("image", "text", "cf"))}
    agree = float(((origin_rm == origin_gr) & active).float().sum() / active.float().sum().clamp_min(1))
    return {"FVE": fve, "L0": float((enc_full > 0).float().sum(-1).mean()),
            "dead": int((~active).sum()),
            "base_R@20": float(m_base["Recall@20"]), "recon_R@20": float(m_rec["Recall@20"]),
            "faith_dR@20": float(m_rec["Recall@20"]) - float(m_base["Recall@20"]),
            "origin_stream_removal": counts_rm,
            "origin_gradient_exact": counts_gr,
            "method_agreement_on_active": agree}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="baby")
    ap.add_argument("--d_sae", type=int, default=1024)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    cfg, ds, model, test_loader = load_frozen("freedom", args.dataset, device)
    s = freedom_streams(model)
    fused, u_all = s["fused"], s["u_all"]
    streams = {"h_img": s["h_img"], "h_txt": s["h_txt"], "cf": s["cf"]}
    pu = torch.from_numpy(np.asarray(ds.train_users)).long().to(device)
    pi = torch.from_numpy(np.asarray(ds.train_items)).long().to(device)

    mde = json.load(open(ROOT / "results/phase0/seed_variance.json"))["summary"]["freedom"]["Recall@20"]["MDE_2std"]
    out_path = OUT / "sae_variants.json"
    existing = json.loads(out_path.read_text()) if out_path.is_file() else {}
    key = f"{args.dataset}_d{args.d_sae}_k{args.k}"
    rec = existing.get(key, {"dataset": args.dataset, "d_sae": args.d_sae, "k": args.k,
                             "MDE": mde, "variants": {}})

    for variant in ("plain", "pred", "mask", "pred+mask"):
        print(f"\n=== variant={variant} ({args.dataset}, d_sae={args.d_sae}, k={args.k}) ===", flush=True)
        sae, mean, scale = train_sae_variant(fused, streams, (pu, pi), u_all, device,
                                             d_sae=args.d_sae, k=args.k, epochs=args.epochs,
                                             variant=variant)
        r = analyze(sae, mean, scale, fused, streams, u_all, test_loader, device)
        rec["variants"][variant] = r
        print(f"  FVE={r['FVE']:.3f} faith_dR@20={r['faith_dR@20']:+.4f} "
              f"(gate {'PASS' if abs(r['faith_dR@20'])<=mde else 'FAIL'}) | "
              f"origin[rm]: img={r['origin_stream_removal']['image']} txt={r['origin_stream_removal']['text']} cf={r['origin_stream_removal']['cf']} | "
              f"origin[grad]: img={r['origin_gradient_exact']['image']} | agree={r['method_agreement_on_active']:.2f}", flush=True)
        existing[key] = rec
        out_path.write_text(json.dumps(existing, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Trainer for :class:`SAE` over arbitrary ``[N, d_in]`` activation tensors.

Deliberately model-agnostic: it never imports a transformer or a recommender.
You hand it a tensor (or DataLoader) of cached activations — residual stream,
CLIP embedding, fused multimodal item embedding, or graph node embedding — and
it returns a trained SAE plus a metrics history.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Union

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

try:
    from .sae import SAE, SAEConfig
except ImportError:  # allow top-level import (when loaded outside the package, e.g. alongside Recsys's `src`)
    from sae import SAE, SAEConfig


@dataclass
class TrainConfig:
    lr: float = 3e-4
    batch_size: int = 4096
    epochs: int = 1
    steps: Optional[int] = None          # if set, overrides epochs (number of optimizer steps)
    device: str = "cuda"
    normalize: bool = True               # mean-subtract + scale to E[||x||]=sqrt(d)
    grad_clip: float = 1.0
    log_every: int = 50
    seed: int = 0
    ckpt_path: Optional[str] = None
    wandb_project: Optional[str] = None  # set to enable wandb logging
    wandb_run: Optional[str] = None


class SAETrainer:
    def __init__(self, sae: SAE, cfg: TrainConfig):
        self.sae = sae
        self.cfg = cfg
        self.opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr)
        self.norm_mean: Optional[torch.Tensor] = None
        self.norm_scale: Optional[float] = None
        self._wandb = None

    # --------------------------------------------------------------- normalize
    def _fit_norm(self, x: torch.Tensor) -> None:
        self.norm_mean = x.mean(0, keepdim=True)
        centered = x - self.norm_mean
        # scale so the expected squared norm equals d (standard SAE preprocessing)
        self.norm_scale = (centered.pow(2).sum(-1).mean().sqrt() / (x.shape[1] ** 0.5)).item()
        self.norm_scale = max(self.norm_scale, 1e-6)

    def _apply_norm(self, x: torch.Tensor) -> torch.Tensor:
        if not self.cfg.normalize or self.norm_mean is None:
            return x
        return (x - self.norm_mean.to(x.device)) / self.norm_scale

    # -------------------------------------------------------------------- data
    def _loader(self, activations: Union[torch.Tensor, DataLoader]) -> DataLoader:
        if isinstance(activations, DataLoader):
            return activations
        ds = TensorDataset(activations)
        return DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=True, drop_last=True)

    # --------------------------------------------------------------------- fit
    def fit(self, activations: Union[torch.Tensor, DataLoader]) -> dict:
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        dev = cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"
        self.sae.to(dev)

        if self.cfg.normalize and isinstance(activations, torch.Tensor):
            self._fit_norm(activations.float())

        if cfg.wandb_project:
            import wandb
            self._wandb = wandb
            wandb.init(project=cfg.wandb_project, name=cfg.wandb_run,
                       config={**self.sae.cfg.to_dict(), **vars(cfg)})

        loader = self._loader(activations)
        history: dict = {"step": [], "recon_loss": [], "l0": [], "fve": [], "dead_frac": []}
        step = 0
        pbar = tqdm(total=cfg.steps or cfg.epochs * len(loader), desc="train SAE")
        done = False
        for _ in range(cfg.epochs if cfg.steps is None else 10_000):
            for (batch,) in loader:
                x = batch.to(dev).float()
                x = self._apply_norm(x)
                out = self.sae.loss(x)
                self.opt.zero_grad(set_to_none=True)
                out["loss"].backward()
                if self.sae.cfg.variant == "relu":
                    self.sae.set_decoder_norm_grad_zero()
                if cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.sae.parameters(), cfg.grad_clip)
                self.opt.step()
                self.sae.normalize_decoder_()

                if step % cfg.log_every == 0:
                    m = self._metrics(x, out)
                    for kk, vv in m.items():
                        history[kk].append(vv)
                    history["step"].append(step)
                    pbar.set_postfix(recon=f"{m['recon_loss']:.4f}", l0=f"{m['l0']:.1f}",
                                     fve=f"{m['fve']:.3f}", dead=f"{m['dead_frac']:.2f}")
                    if self._wandb:
                        self._wandb.log(m, step=step)

                step += 1
                pbar.update(1)
                if cfg.steps is not None and step >= cfg.steps:
                    done = True
                    break
            if done:
                break
        pbar.close()

        if cfg.ckpt_path:
            self.save(cfg.ckpt_path)
        if self._wandb:
            self._wandb.finish()
        return history

    # ----------------------------------------------------------------- metrics
    @torch.no_grad()
    def _metrics(self, x: torch.Tensor, out: dict) -> dict:
        acts, recon = out["acts"], out["recon"]
        l0 = (acts > 0).float().sum(-1).mean().item()
        resid_var = (x - recon).pow(2).sum(-1).mean()
        total_var = (x - x.mean(0, keepdim=True)).pow(2).sum(-1).mean()
        fve = (1 - resid_var / total_var.clamp_min(1e-8)).item()  # fraction of variance explained
        return {
            "recon_loss": out["recon_loss"].item(),
            "l0": l0,
            "fve": fve,
            "dead_frac": self.sae.num_dead / self.sae.cfg.d_sae,
        }

    # -------------------------------------------------------------------- save
    def save(self, path: str) -> None:
        torch.save({
            "sae_state": self.sae.state_dict(),
            "sae_cfg": self.sae.cfg.to_dict(),
            "norm_mean": self.norm_mean,
            "norm_scale": self.norm_scale,
        }, path)

    @staticmethod
    def load(path: str, map_location: str = "cpu"):
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        sae = SAE(SAEConfig(**ckpt["sae_cfg"]))
        sae.load_state_dict(ckpt["sae_state"])
        return sae, ckpt

"""GUME + Behavior-Aligned Image (BAI) / Co-purchase-Supervised Image (CSI).

Constructive follow-up to the measurement paper. Analysis (results/bai/) established:
  * raw image cosine ~= chance at predicting co-purchase (held-out AUC 0.54), which is what
    every graph MM-rec uses to route image -> image ignored;
  * a projection trained DIRECTLY on co-purchase pairs generalizes (held-out AUC 0.74, image >
    text) and adds +0.0099 AUC beyond GUME's FULL representation -> real, complementary signal;
  * the recommender scores user-item, so image needs its OWN user-side factor to be used.

Three toggleable mechanisms on top of GUME (imported verbatim from read-only Recsys; baseline is
byte-identical when all flags off):
  bai_channel : SEPARATE image scoring channel  score += u_img . z_img  (the principled method) --
                z_img = co-purchase-supervised item-image embedding, u_img = learned user-image
                embedding (init ~0 so we start at GUME and grow only if image helps).
  bai_enable  : image RESIDUAL added to the item embedding (weaker; lacks a user-side factor).
  bai_graph_mode='cooc' : rebuild GUME's image graph along co-occurring edges only.
  bai_cop     : direct co-purchase metric supervision of the image projection (the key ingredient).
  bai_reg     : L2 penalty on the residual (residual mode only).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.gume import GUME, build_sim, build_knn_normalized_graph


class GUME_BAI(GUME):
    def __init__(self, config, n_users, n_items, norm_adj=None,
                 v_feat=None, t_feat=None, train_user_idx=None, train_item_idx=None):
        super().__init__(config, n_users, n_items, norm_adj=norm_adj,
                         v_feat=v_feat, t_feat=t_feat,
                         train_user_idx=train_user_idx, train_item_idx=train_item_idx)
        self.bai_enable = bool(config.get("bai_enable", False))
        self.bai_channel = bool(config.get("bai_channel", False))
        self.bai_alpha = float(config.get("bai_alpha", 1.0))
        self.bai_align = float(config.get("bai_align", 0.0))
        self.bai_align_temp = float(config.get("bai_align_temp", 0.2))
        self.bai_reg = float(config.get("bai_reg", 0.0))
        self.bai_cop = float(config.get("bai_cop", 0.0))
        self.bai_chan_reg = float(config.get("bai_chan_reg", 0.0))
        self.bai_chan_scale = float(config.get("bai_chan_scale", 1.0))
        self.bai_loo_weight = float(config.get("bai_loo_weight", 1.0))
        self.bai_cotrain = bool(config.get("bai_cotrain", False))
        self._cur = None
        # ---- objective upgrade: in-batch Sampled-Softmax + popularity (logQ) debiasing ----
        # Diagnosis: BPR with ONE uniform-random negative is too easy -> coarse CF/popularity
        # separates it, so gradients never reward fine multimodal discrimination (why content is
        # unused). Replacing BPR with an in-batch softmax over ~train_batch_size negatives forces
        # fine-grained ranking; the logQ term debiases the popularity of in-batch negatives.
        self.bai_loss = str(config.get("bai_loss", "bpr"))
        self.ssm_temp = float(config.get("ssm_temp", 0.15))
        self.ssm_debias = float(config.get("ssm_debias", 1.0))
        self.ssm_norm = bool(config.get("ssm_norm", True))
        # popularity prior added to BOTH train logits and eval scores; restores the recall-friendly
        # popularity signal that cosine normalization strips out of the embedding norms.
        self.ssm_popbias = float(config.get("ssm_popbias", 0.0))
        # AUX mode: keep GUME's full BPR+InfoNCE objective UNCHANGED and ADD a weighted in-batch
        # softmax ranking term (sharpens fine-grained discrimination without destabilizing the
        # tuned base). Eval is GUME's default dot-product. ssm_aux is the term weight.
        self.ssm_aux = float(config.get("ssm_aux", 0.0))
        if self.ssm_aux > 0.0 and self.bai_loss != "ssm":
            pop = np.clip(np.asarray(self.interaction_matrix.sum(0)).flatten().astype(np.float32), 1.0, None)
            self.register_buffer("item_logpop", torch.log(torch.from_numpy(pop)), persistent=False)
        # DISTILLATION: train ONE GUME student to mimic the diverse ensemble's in-batch ranking
        # (BPR + lambda*KL) -> a single 1x-cost model that absorbs the ensemble gain.
        self.distill_weight = float(config.get("distill_weight", 0.0))
        self.distill_temp = float(config.get("distill_temp", 1.0))
        tpath = config.get("distill_teacher", None)
        if self.distill_weight > 0.0 and tpath:
            self.register_buffer("teacher", torch.from_numpy(np.load(tpath).astype(np.float32)),
                                 persistent=False)
        if self.bai_loss == "ssm":
            pop = np.asarray(self.interaction_matrix.sum(0)).flatten().astype(np.float32)
            pop = np.clip(pop, 1.0, None)
            self.register_buffer("item_logpop", torch.log(torch.from_numpy(pop)), persistent=False)
        gate_bias = float(config.get("bai_gate_bias", -2.0))
        d = self.embedding_dim
        use_img = self.bai_enable or self.bai_channel

        if use_img:
            # optional external image features (e.g. CNN+DINOv2+SigLIP2 ensemble) as the
            # projection input; default uses GUME's own v_feat (the dataset CNN features).
            img_npy = config.get("bai_img_npy", None)
            if img_npy:
                src = np.load(img_npy).astype(np.float32)
                self.register_buffer("bai_img_src", torch.from_numpy(src), persistent=False)
                vdim = int(src.shape[1])
            else:
                self.bai_img_src = None
                vdim = int(self.v_feat.shape[1])
            # projection of the RAW (frozen) image content -> behavior space
            self.bai_img_proj = nn.Sequential(
                nn.Linear(vdim, d), nn.GELU(), nn.Linear(d, d))
            if self.bai_cop > 0.0:
                self.register_buffer("cop_pairs", self._build_cop_pairs(), persistent=False)

        if self.bai_channel:
            self.bai_user_img_mode = str(config.get("bai_user_img_mode", "content"))
            if self.bai_user_img_mode == "loo":
                # LOO mode uses a FIXED channel scale (single interpretable knob); z trained by the
                # leave-one-out ranking loss. cs learnable in other modes.
                self.bai_cs = float(config.get("bai_chan_scale", 1.0))
            else:
                # learnable channel scale, small init -> starts near GUME, grows only if image helps
                self.bai_cs = nn.Parameter(torch.full((1,), float(config.get("bai_chan_scale", 0.3))))
            if self.bai_user_img_mode == "free":
                # free per-user image factor (overfits on Baby); init ~0
                self.bai_user_img = nn.Embedding(self.n_users, d)
                nn.init.normal_(self.bai_user_img.weight, std=1e-2)
            elif self.bai_user_img_mode == "loo":
                # LEAVE-ONE-OUT content profile. The profile of user u = mean of z_img over u's
                # train items; at TRAIN time the positive item is EXCLUDED from its own profile
                # (fixes the self-inclusion leak that sank the plain 'content' channel). Needs the
                # basket SUM (binary R) and per-user counts; profile is rebuilt from the *current*
                # z each step so the projection is supervised end-to-end.
                import scipy.sparse as sp
                R = (self.interaction_matrix.tocsr() > 0).astype(np.float32)
                self.register_buffer("bai_R", self._sp2t(R.tocoo()), persistent=False)
                n_u = np.asarray(R.sum(1)).flatten().astype(np.float32)
                self.register_buffer("bai_n", torch.from_numpy(n_u).clamp(min=1.0), persistent=False)
            else:
                # CONTENT-derived user image profile = row-normalized mean of the user's train
                # items' z_img. No free per-user params -> cannot overfit per-user.
                import scipy.sparse as sp
                R = self.interaction_matrix.tocsr().astype(np.float32)
                rs = np.asarray(R.sum(1)).flatten(); rs[rs == 0] = 1.0
                Rn = sp.diags(1.0 / rs).dot(R).tocoo()
                self.register_buffer("bai_Rn", self._sp2t(Rn), persistent=False)

        if self.bai_enable:
            gate_lin = nn.Linear(d, d)
            nn.init.zeros_(gate_lin.weight)
            nn.init.constant_(gate_lin.bias, gate_bias)
            self.bai_gate = nn.Sequential(gate_lin, nn.Sigmoid())
            self.bai_scale = nn.Parameter(
                torch.full((1,), float(config.get("bai_scale_init", 0.1))))

        # Behavior-aligned image graph (independent of residual/channel): mask GUME's visual-kNN
        # image graph to co-occurring item pairs and renormalize.
        self.bai_graph_mode = str(config.get("bai_graph_mode", "none"))
        if self.bai_graph_mode == "cooc":
            R = self.interaction_matrix.tocsr().astype(np.float32)
            C = (R.T @ R).tocoo()
            cooc = torch.zeros(self.n_items, self.n_items)
            cooc[torch.from_numpy(C.row).long(), torch.from_numpy(C.col).long()] = torch.from_numpy(C.data)
            cooc.fill_diagonal_(0.0)
            vsim = build_sim(self.v_feat).cpu()
            masked = vsim * (cooc > 0).float()
            new_adj, _ = build_knn_normalized_graph(masked, topk=self.knn_k, norm_type='sym')
            self.image_original_adj = new_adj.to(self.image_original_adj.device)

    # ---- co-purchase supervision ----
    def _build_cop_pairs(self):
        from collections import defaultdict
        coo = self.interaction_matrix.tocoo()
        ub = defaultdict(list)
        for u, i in zip(coo.row.tolist(), coo.col.tolist()):
            ub[u].append(i)
        pairs = set()
        for items in ub.values():
            items = list(set(items))
            for a in range(len(items)):
                for b in range(a + 1, len(items)):
                    x, y = items[a], items[b]
                    pairs.add((x, y) if x < y else (y, x))
        return torch.tensor(sorted(pairs), dtype=torch.long)

    def _img_proj_norm(self):
        src = self.bai_img_src if getattr(self, "bai_img_src", None) is not None else self.v_feat
        return F.normalize(self.bai_img_proj(src), dim=1)

    def _cop_loss(self, n_sample=4096):
        z = self._img_proj_norm()
        dev = z.device
        idx = torch.randint(0, self.cop_pairs.shape[0], (n_sample,), device=dev)
        p = self.cop_pairs.to(dev)[idx]
        na = torch.randint(0, self.n_items, (n_sample,), device=dev)
        nb = torch.randint(0, self.n_items, (n_sample,), device=dev)
        pos = (z[p[:, 0]] * z[p[:, 1]]).sum(1)
        neg = (z[na] * z[nb]).sum(1)
        return -(F.logsigmoid(pos - neg)).mean()

    # ---- residual (mechanism: in-space addition; lacks a user-side image factor) ----
    def _img_residual(self):
        z = self._img_proj_norm()
        g = self.bai_gate(self.item_id_embedding.weight)
        return self.bai_alpha * self.bai_scale * g * z

    def forward(self, adj, train=False):
        out = super().forward(adj, train=train)
        if not (self.bai_enable or self.bai_channel):
            return out
        if train:
            all_embeds, e2, e3 = out
        else:
            all_embeds = out

        if self.bai_channel and self.bai_user_img_mode == "loo":
            # LEAVE-ONE-OUT profile channel. At TRAIN time the channel is NOT folded into the
            # embeddings (it is supervised separately, leak-free, in calculate_loss); at EVAL the
            # full user-image profile (mean of the user's train-item z, normalized) is concatenated
            # so the u.item scoring picks up  cs * cos(profile_u, z_i).  Test items are never in the
            # train profile -> no leak at eval.
            if not train:
                z0 = self._img_proj_norm()
                prof = torch.sparse.mm(self.bai_R, z0) / self.bai_n.unsqueeze(1)
                u_img = self.bai_cs * F.normalize(prof, dim=1)
                users = torch.cat([all_embeds[:self.n_users], u_img], dim=1)
                items = torch.cat([all_embeds[self.n_users:], z0], dim=1)
                all_embeds = torch.cat([users, items], dim=0)
        elif self.bai_channel:
            # SEPARATE image channel: concatenate u_img to users and z_img to items so the
            # existing u.item scoring picks up an additive u_img . z_img term.
            z0 = self._img_proj_norm()                                     # [n_items, d]
            if self.bai_user_img_mode == "free":
                u0 = self.bai_user_img.weight                              # [n_users, d]
            else:
                u0 = torch.sparse.mm(self.bai_Rn, z0)                      # content-derived profile
            z_img = self.bai_cs * z0
            u_img = self.bai_cs * u0
            users = torch.cat([all_embeds[:self.n_users], u_img], dim=1)
            items = torch.cat([all_embeds[self.n_users:], z_img], dim=1)
            all_embeds = torch.cat([users, items], dim=0)
        elif self.bai_enable:
            res = self._img_residual()
            pad = torch.zeros(self.n_users, self.embedding_dim,
                              device=res.device, dtype=res.dtype)
            all_embeds = all_embeds + torch.cat([pad, res], dim=0)

        if train:
            return all_embeds, e2, e3
        return all_embeds

    def _align_loss(self, users, pos_items):
        z = self._img_proj_norm()
        zp = F.normalize(z[pos_items], dim=1)
        item_id = F.normalize(self.item_id_embedding.weight[pos_items], dim=1).detach()
        logits = (zp @ item_id.t()) / self.bai_align_temp
        labels = torch.arange(zp.shape[0], device=zp.device)
        return F.cross_entropy(logits, labels)

    def _loo_bpr(self, interaction):
        """Leave-one-out user-image-profile ranking loss (leak-free).
        For (u, pos, neg): profile_u EXCLUDES pos from its own basket; score = cs*cos(prof, z).
        Only users with >=2 train items contribute (LOO profile undefined for singletons)."""
        users = interaction["user"]; pos = interaction["pos_item"]; neg = interaction["neg_item"]
        z = self._img_proj_norm()                                  # [n_items, d], unit rows
        S = torch.sparse.mm(self.bai_R, z)                         # [n_users, d] basket sum
        nu = self.bai_n[users].unsqueeze(1)                        # [B, 1]
        keep = (nu.squeeze(1) >= 2)
        if keep.sum() < 2:
            return z.new_zeros(())
        Su = S[users]
        prof_loo = F.normalize((Su - z[pos]) / (nu - 1.0).clamp(min=1.0), dim=1)
        prof_full = F.normalize(Su / nu, dim=1)                    # neg item (~never in basket)
        pos_s = self.bai_cs * (prof_loo * z[pos]).sum(1)
        neg_s = self.bai_cs * (prof_full * z[neg]).sum(1)
        return -(F.logsigmoid((pos_s - neg_s)[keep])).mean()

    def _ssm_loss(self, users, pos_items, neg_items):
        """In-batch Sampled-Softmax: every other positive in the batch is a negative for this
        user (~train_batch_size-1 negatives), with a logQ popularity-debias term. Diagonal is the
        true positive. Duplicate items in the batch are masked off so they are not false negatives.
        Keeps GUME's L2 regularizer unchanged so only the ranking objective changes."""
        u_idx, p_idx, n_idx = self._cur
        B = users.shape[0]
        if self.ssm_norm:
            u = F.normalize(users, dim=1); it = F.normalize(pos_items, dim=1)
        else:
            u, it = users, pos_items
        logits = (u @ it.t()) / self.ssm_temp                          # [B, B]
        if self.ssm_debias > 0:
            logits = logits - self.ssm_debias * self.item_logpop[p_idx].unsqueeze(0)
        if self.ssm_popbias > 0:
            logits = logits + self.ssm_popbias * self.item_logpop[p_idx].unsqueeze(0)
        # mask off-diagonal cells whose candidate item equals the row's positive item
        same = (p_idx.unsqueeze(0) == p_idx.unsqueeze(1))
        eye = torch.eye(B, dtype=torch.bool, device=users.device)
        logits = logits.masked_fill(same & ~eye, float("-inf"))
        labels = torch.arange(B, device=users.device)
        ssm = F.cross_entropy(logits, labels)
        reg = self.reg_weight_1 * (self.sq_sum(users) + self.sq_sum(pos_items)
                                   + self.sq_sum(neg_items)) / self.batch_size
        return ssm, reg

    def full_sort_predict(self, interaction):
        # SSM is trained in normalized-cosine geometry -> score the same way at eval for consistency.
        if getattr(self, "bai_loss", "bpr") == "ssm" and self.ssm_norm:
            user = interaction["user"]
            all_embeds = self.forward(self.gume_norm_adj)
            ue, ie = torch.split(all_embeds, [self.n_users, self.n_items], dim=0)
            scores = F.normalize(ue[user], dim=1) @ F.normalize(ie, dim=1).t()
            if self.ssm_popbias > 0:
                scores = scores + self.ssm_popbias * self.item_logpop.unsqueeze(0)
            return scores
        return super().full_sort_predict(interaction)

    def bpr_loss(self, users, pos_items, neg_items):
        """GUME's BPR, but for CO-TRAINED loo mode fold the leak-free image-profile channel into
        the pos/neg scores so the base embeddings co-adapt (gradient flows into base AND z).
        _cur (set by calculate_loss) carries the batch indices; None -> identical to GUME."""
        if getattr(self, "bai_loss", "bpr") == "ssm" and self._cur is not None:
            return self._ssm_loss(users, pos_items, neg_items)
        mf, reg = super().bpr_loss(users, pos_items, neg_items)
        # AUX: keep GUME's BPR + add an in-batch softmax ranking term (dot-product geometry).
        if getattr(self, "ssm_aux", 0.0) > 0.0 and self._cur is not None:
            u_idx, p_idx, _ = self._cur
            B = users.shape[0]
            logits = (users @ pos_items.t()) / self.ssm_temp
            if self.ssm_debias > 0:
                logits = logits - self.ssm_debias * self.item_logpop[p_idx].unsqueeze(0)
            same = (p_idx.unsqueeze(0) == p_idx.unsqueeze(1))
            eye = torch.eye(B, dtype=torch.bool, device=users.device)
            logits = logits.masked_fill(same & ~eye, float("-inf"))
            aux = F.cross_entropy(logits, torch.arange(B, device=users.device))
            return mf + self.ssm_aux * aux, reg
        # DISTILL: match the teacher ensemble's in-batch ranking (KL on softmax over batch items)
        if getattr(self, "distill_weight", 0.0) > 0.0 and self._cur is not None:
            u_idx, p_idx, _ = self._cur
            s_logit = (users @ pos_items.t()) / self.distill_temp
            t_logit = self.teacher[u_idx][:, p_idx] / self.distill_temp
            distill = F.kl_div(F.log_softmax(s_logit, dim=1),
                               F.softmax(t_logit, dim=1), reduction="batchmean")
            return mf + self.distill_weight * distill, reg
        if self._cur is None:
            return mf, reg
        u_idx, p_idx, n_idx = self._cur
        z = self._img_proj_norm()
        S = torch.sparse.mm(self.bai_R, z)
        nu = self.bai_n[u_idx].unsqueeze(1)
        prof_loo = F.normalize((S[u_idx] - z[p_idx]) / (nu - 1.0).clamp(min=1.0), dim=1)
        prof_full = F.normalize(S[u_idx] / nu.clamp(min=1.0), dim=1)
        ch_pos = self.bai_cs * (prof_loo * z[p_idx]).sum(1)
        ch_neg = self.bai_cs * (prof_full * z[n_idx]).sum(1)
        pos_scores = (users * pos_items).sum(1) + ch_pos
        neg_scores = (users * neg_items).sum(1) + ch_neg
        return -(F.logsigmoid(pos_scores - neg_scores)).mean(), reg

    def calculate_loss(self, interaction):
        cotrain = (self.bai_channel and getattr(self, "bai_user_img_mode", "") == "loo"
                   and self.bai_cotrain)
        if (cotrain or getattr(self, "bai_loss", "bpr") == "ssm" or getattr(self, "ssm_aux", 0.0) > 0
                or getattr(self, "distill_weight", 0.0) > 0):
            self._cur = (interaction["user"], interaction["pos_item"], interaction["neg_item"])
        loss = super().calculate_loss(interaction)
        self._cur = None
        if (self.bai_channel and getattr(self, "bai_user_img_mode", "") == "loo"
                and not self.bai_cotrain):
            loss = loss + self.bai_loo_weight * self._loo_bpr(interaction)
        if self.bai_enable or self.bai_channel:
            if self.bai_cop > 0.0:
                loss = loss + self.bai_cop * self._cop_loss()
            if self.bai_align > 0.0:
                loss = loss + self.bai_align * self._align_loss(
                    interaction["user"], interaction["pos_item"])
            if self.bai_reg > 0.0 and self.bai_enable:
                loss = loss + self.bai_reg * (self._img_residual() ** 2).mean()
            if (self.bai_channel and self.bai_chan_reg > 0.0
                    and getattr(self, "bai_user_img_mode", "content") == "free"):
                u = self.bai_user_img(interaction["user"])
                loss = loss + self.bai_chan_reg * u.pow(2).sum() / u.shape[0]
        return loss

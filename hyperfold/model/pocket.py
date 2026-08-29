"""Hyper-Fold-Pocket: set-prediction pocket detection head on the
Hyper-Fold backbone.

Pipeline (paper decomposition):
    Hyper-Fold backbone (6 Fold-Conv blocks) -> BiLevelNeck (two readout
    levels projected to d_model; fused residue features X_res, mask keys
    K_emb, bi-level memory M_emb) -> Pocket Proposal Network (per-residue
    pocketness probe) -> top-N_q structure-anchored queries (content =
    fused residue features, position = CA coordinate via the 3D Fourier
    positional encoding) -> 4-layer transformer decoder over the bi-level
    memory -> classification head (pocket / no-object) + mask head (inner
    product of the decoded query with K_emb).

Training extras: contrastive denoising (CDN) queries decoded in a separate,
attention-isolated pass and excluded from Hungarian matching, plus a
residue-level BCE loss on the pocketness scores.

No ESM / 1D information is used anywhere: node input is the amino-acid
type (21 classes) plus CA coordinates; the UniProt sequence axis is used
only to index the output masks.
"""
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .backbone import HyperFold
from .neck import BiLevelNeck
from .position import FourierPositionalEncoding3D
from .decoder import TransformerDecoder, TransformerDecoderLayer
from .losses import SetCriterion, sigmoid_ce_loss_jit, dice_loss_jit


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN), as in DETR."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class PocketProposalNetwork(nn.Module):
    """Per-residue pocketness probe: a linear head scoring every residue of
    the fused features X_res; the top-N_q residues anchor the decoder
    queries."""

    def __init__(self, d_model: int):
        super().__init__()
        self.pocketness = nn.Linear(d_model, 1)

    def forward(self, X_res: Tensor) -> Tensor:
        """X_res: [N, L, d_model] -> pocketness scores [N, L]."""
        return self.pocketness(X_res).squeeze(-1)


class HyperFoldPocket(nn.Module):
    """Hyper-Fold-Pocket: structure-anchored set predictor for pockets.

    Hyper-Fold backbone -> BiLevelNeck -> PPN pocketness scores -> top-N_q
    structure-anchored queries (content = fused residue features X_res,
    position = CA coordinate via 3D-PE) -> 4-layer transformer decoder over
    the bi-level memory M_emb -> classification head + mask head (inner
    product with the mask keys K_emb).
    """

    def __init__(self, model_conf):
        super().__init__()
        self._model_conf = model_conf
        d_model = model_conf.node_embed_size
        self.num_queries = _conf_get(model_conf, 'num_queries', 50)
        self.dn_per_gt = _conf_get(model_conf, 'dn_per_gt', 2)
        self.dn_noise_feat = _conf_get(model_conf, 'dn_noise_feat', 0.5)
        self.dn_noise_xyz = _conf_get(model_conf, 'dn_noise_xyz', 2.0)
        self.max_dn = _conf_get(model_conf, 'max_dn', 32)

        readout_blocks = tuple(_conf_get(model_conf, 'readout_blocks', (3, 6)))
        mask_readout_blocks = _conf_get(model_conf, 'mask_readout_blocks', None)
        if mask_readout_blocks is not None:
            mask_readout_blocks = tuple(mask_readout_blocks)
        conv_type = _conf_get(model_conf, 'conv_type', 'foldconv')
        if conv_type == 'cdconv':
            conv_type = 'foldconv'  # legacy config key (old train_args.json)
        self.backbone = HyperFold(
            readout_blocks=readout_blocks,
            mask_readout_blocks=mask_readout_blocks,
            conv_type=conv_type,
            k_small=_conf_get(model_conf, 'k_small', 8))
        self.neck = BiLevelNeck(
            width=self.backbone.width, d_model=d_model,
            readout_blocks=readout_blocks,
            mask_readout_blocks=mask_readout_blocks)
        self.pos_enc = FourierPositionalEncoding3D(d_model)
        self.ppn = PocketProposalNetwork(d_model)
        self.pad_query = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.pad_query, std=0.02)

        decoder_layer = TransformerDecoderLayer(
            d_model, model_conf.no_heads, model_conf.dim_feedforward,
            model_conf.dropout, "relu", normalize_before=False)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer, _conf_get(model_conf, 'dec_layers', 4),
            decoder_norm, return_intermediate=True)
        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.cls_embed = MLP(d_model, d_model, model_conf.num_classes + 1, 2)
        self.mask_embed = MLP(d_model, d_model, d_model, 3)
        # contrastive denoising: heavily-noised no-object copies per GT
        self.dn_neg_per_gt = _conf_get(model_conf, 'dn_neg_per_gt', 0)

    def _select_queries(self, X_res, scores, xyz_full, eligible):
        """Top-K eligible residues per protein -> content, anchors."""
        N, L, C = X_res.shape
        K = self.num_queries
        sel_scores = scores.masked_fill(~eligible, float('-inf'))
        top_idx = sel_scores.topk(min(K, L), dim=1).indices  # [N, K]
        n_eligible = eligible.sum(dim=1)  # [N]
        content = self.pad_query.expand(N, K, C).clone()
        anchors = torch.zeros(N, K, 3, device=X_res.device)
        for i in range(N):
            k = int(min(n_eligible[i].item(), K))
            if k == 0:
                continue
            idx = top_idx[i, :k]
            content[i, :k] = X_res[i, idx]
            anchors[i, :k] = xyz_full[i, idx]
        return content, anchors

    def _decode(self, content, query_pos, memory, mem_pad, mem_pos):
        hs = self.decoder(content.transpose(0, 1), memory.transpose(0, 1),
                          memory_key_padding_mask=mem_pad,
                          pos=mem_pos.transpose(0, 1),
                          query_pos=query_pos.transpose(0, 1))
        return hs.transpose(1, 2)  # [layers, N, Q, C]

    def _heads(self, hs, K_emb):
        logits = self.cls_embed(hs)  # [layers, N, Q, ncls+1]
        mask_q = self.mask_embed(hs)
        masks = torch.einsum('dbqc,blc->dbql', mask_q, K_emb)
        return logits, masks

    def build_cdn_queries(self, X_res, xyz_full, mapped_mask, targets):
        """Contrastive denoising (CDN) queries built from the GT pockets.

        Per GT pocket, the content is the GT-mask-pooled fused residue
        feature and the anchor the GT CA centroid. Each GT contributes
        ``dn_per_gt`` copies: 1 clean (exact pooled content / centroid) and
        the rest lightly noised (``dn_noise_feat`` / ``dn_noise_xyz``), all
        supervised against their source pocket. With ``dn_neg_per_gt`` > 0,
        extra heavily-noised contrastive copies (2x feature / 3x coordinate
        noise) are appended and labeled no-object. Denoising queries are
        decoded in a separate, attention-isolated pass (no cross-talk with
        the detection queries) and skip Hungarian matching: they align 1:1
        with their source GT.

        Returns content, anchors, src index lists (GT index per query) and
        neg flags (1 = contrastive no-object copy).
        """
        N, _, C = X_res.shape
        Qdn = self.max_dn
        content = self.pad_query.expand(N, Qdn, C).clone()
        anchors = torch.zeros(N, Qdn, 3, device=X_res.device)
        src_list, neg_list = [], []
        for i in range(N):
            t = targets[i]
            gt_masks = t['pocket_masks'].to(X_res.device).float()  # [P, L']
            Lp = gt_masks.shape[1]
            f_i = X_res[i, :Lp]
            m_i = mapped_mask[i, :Lp].float()
            xyz_i = xyz_full[i, :Lp]
            srcs, negs = [], []
            for j in range(gt_masks.shape[0]):
                w = gt_masks[j] * m_i
                if w.sum() < 1:
                    continue
                w = w / w.sum()
                c = (w.unsqueeze(-1) * f_i).sum(0)
                a = (w.unsqueeze(-1) * xyz_i).sum(0)
                for noisy in range(self.dn_per_gt):
                    if len(srcs) >= Qdn:
                        break
                    q = len(srcs)
                    if noisy > 0:
                        content[i, q] = c + torch.randn_like(c) * self.dn_noise_feat
                        anchors[i, q] = a + torch.randn_like(a) * self.dn_noise_xyz
                    else:
                        content[i, q] = c
                        anchors[i, q] = a
                    srcs.append(j)
                    negs.append(0)
                # contrastive negative copies: large noise -> no-object
                for _ in range(self.dn_neg_per_gt):
                    if len(srcs) >= Qdn:
                        break
                    q = len(srcs)
                    content[i, q] = c + torch.randn_like(c) * self.dn_noise_feat * 2
                    anchors[i, q] = a + torch.randn_like(a) * self.dn_noise_xyz * 3
                    srcs.append(j)
                    negs.append(1)
                if len(srcs) >= Qdn:
                    break
            src_list.append(srcs)
            neg_list.append(negs)
        return content, anchors, src_list, neg_list

    def forward(self, input_feats, targets=None):
        L = input_feats['aatype'].shape[1]
        res_mask = input_feats['res_mask'].bool()  # [N, L], True = valid
        hidden, xyz_full, mapped_mask = self.backbone(input_feats, L)
        X_res, K_emb, M_emb, mem_pad = self.neck(
            hidden, xyz_full, mapped_mask, res_mask, L)

        n_lv = len(self.neck.readout_blocks)
        mem_pos = self.pos_enc(xyz_full).repeat(1, n_lv, 1)

        scores = self.ppn(X_res)  # [N, L] pocketness scores
        eligible = mapped_mask & res_mask
        content, anchors = self._select_queries(X_res, scores, xyz_full,
                                                eligible)
        query_pos = self.pos_enc(anchors)

        hs = self._decode(content, query_pos, M_emb, mem_pad, mem_pos)
        logits, masks = self._heads(hs, K_emb)

        out = {'pred_logits': logits[-1], 'pred_masks': masks[-1],
               'residue_scores': scores,
               'aux_outputs': [{'pred_logits': a, 'pred_masks': b}
                               for a, b in zip(logits[:-1], masks[:-1])]}

        if self.training and targets is not None:
            dn_content, dn_anchors, dn_src, dn_neg = self.build_cdn_queries(
                X_res, xyz_full, mapped_mask, targets)
            dn_pos = self.pos_enc(dn_anchors)
            hs_dn = self._decode(dn_content, dn_pos, M_emb, mem_pad, mem_pos)
            logits_dn, masks_dn = self._heads(hs_dn, K_emb)
            out['dn_outputs'] = {'pred_logits': logits_dn[-1],
                                 'pred_masks': masks_dn[-1],
                                 'src': dn_src, 'neg': dn_neg}
        return out

    @torch.no_grad()
    def post_process(self, outputs, lengths, mask_thresh=0.5,
                     return_numpy=False):
        pred_logits = outputs['pred_logits']
        pred_masks = outputs['pred_masks']
        final_outputs = []
        for i, num_res in enumerate(lengths):
            pred_prob = pred_logits[i].softmax(-1)
            scores, labels = pred_prob[:, :-1].max(-1)
            mask_probs = pred_masks[i][:, :num_res].sigmoid()
            masks = (mask_probs > mask_thresh).float()
            # score = cls prob x mean mask prob within the binarized mask
            scores *= (masks * mask_probs).sum(-1) / (masks.sum(-1) + 1e-6)
            _, indices = scores.sort(descending=True)
            entry = {'scores': scores[indices], 'labels': labels[indices],
                     'pocket_masks': masks[indices]}
            if return_numpy:
                entry = {k: v.cpu().numpy() for k, v in entry.items()}
            final_outputs.append(entry)
        return final_outputs


class PocketSetCriterion(SetCriterion):
    """SetCriterion + residue-level pocketness BCE + CDN query losses.

    Adds, on top of the Hungarian-matched classification / mask / dice
    losses of :class:`SetCriterion`:
      * ``loss_res``: residue-level BCE on the PPN pocketness scores;
      * ``loss_cls_dn`` / ``loss_mask_dn`` / ``loss_dice_dn``: losses of the
        contrastive denoising queries, aligned 1:1 with their source GT
        pocket (no Hungarian matching); contrastive negatives are labeled
        no-object and excluded from the mask/dice terms.
    """

    def forward(self, outputs, targets):
        loss_dict = super().forward(outputs, targets)

        num_masks = max(float(sum(len(t['labels']) for t in targets)), 1.0)

        # residue-level BCE on the pocketness scores
        res_loss = 0.0
        for scores, t in zip(outputs['residue_scores'], targets):
            valid = t['res_mask']
            gt = (t['pocket_masks'].sum(0) > 0).float().to(scores.device)
            res_loss = res_loss + F.binary_cross_entropy_with_logits(
                scores[:len(valid)][valid], gt[valid])
        loss_dict['loss_res'] = res_loss / len(targets)

        # denoising queries: aligned 1:1 with their source GT pocket
        if 'dn_outputs' in outputs:
            dn = outputs['dn_outputs']
            negs_all = dn.get('neg') or [[] for _ in dn['src']]
            cls_loss, mask_loss, dice_loss = 0.0, 0.0, 0.0
            for i, (t, srcs) in enumerate(zip(targets, dn['src'])):
                if not srcs:
                    continue
                negs = negs_all[i]
                idx = torch.tensor(srcs, dtype=torch.long)
                logits = dn['pred_logits'][i, :len(srcs)]  # [Qdn, 2]
                # positives -> pocket class 0; CDN negatives -> no-object
                cls_target = torch.tensor(negs, dtype=torch.long,
                                          device=logits.device) \
                    * self.num_classes
                cls_loss = cls_loss + F.cross_entropy(logits, cls_target)
                valid = t['res_mask']
                pos = [q for q, ng in enumerate(negs) if ng == 0]
                if pos:
                    pos_t = torch.tensor(pos, dtype=torch.long)
                    pred = dn['pred_masks'][i, pos_t][:, :len(valid)][:, valid]
                    gt = t['pocket_masks'].to(pred.device)[
                        idx.to(pred.device)[pos_t]][:, valid]
                    mask_loss = mask_loss + sigmoid_ce_loss_jit(
                        pred, gt, num_masks)
                    dice_loss = dice_loss + dice_loss_jit(pred, gt, num_masks)
            n = max(sum(len(s) for s in dn['src']), 1)
            loss_dict['loss_cls_dn'] = cls_loss / n
            loss_dict['loss_mask_dn'] = mask_loss
            loss_dict['loss_dice_dn'] = dice_loss
        return loss_dict


def _conf_get(model_conf, key, default):
    """Attribute-style config lookup with a default (SimpleNamespace-like)."""
    return getattr(model_conf, key, default)

"""Losses for Hyper-Fold-Pocket.

DETR-style Hungarian matching and set criterion over pocket masks, the
JIT-scripted dice / sigmoid cross-entropy losses they rely on, and the
convex-hull pocket-center computation used for DCC/DCA evaluation.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull, Delaunay

from ..utils.misc import is_dist_avail_and_initialized


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    len = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / len


batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def hull_center(hull):
    """Compute the convex-hull center of a predicted pocket mask.

    Used for DCC/DCA evaluation of predicted pockets.

    Parameters
    ----------
    hull: scipy.spatial.ConvexHull
        Convex hull to compute the center of.

    Returns
    -------
    numpy.ndarray
        Convex hull center of mass.
    """

    hull_com = np.zeros(3)
    tetras = Delaunay(hull.points[hull.vertices])

    for i in range(len(tetras.simplices)):
        tetra_verts = tetras.points[tetras.simplices][i]

        a, b, c, d = tetra_verts
        a, b, c = a - d, b - d, c - d
        tetra_vol = np.abs(np.linalg.det([a, b, c])) / 6

        tetra_com = np.mean(tetra_verts, axis=0)

        hull_com += tetra_com * tetra_vol

    hull_com = hull_com / hull.volume
    return hull_com


def calc_center(protein, masks, methods=["hull", "geo"]):
    """Calculate the center of a protein pocket.

    Parameters
    ----------
    protein: hyperfold.data.graph.ProteinGraph or AtomLevelProtein
        Protein to calculate the center of; ``protein[i].node_position``
        gives the atom coordinates of residue i.
    masks: numpy.ndarray
        Pocket masks to calculate the center of.
    methods: list of str
        Methods to use for calculating the center. Options are "hull" and "geo".

    Returns
    -------
    dict of numpy.ndarray
        Center of the protein pocket under each requested method.
    """
    hull_centers = []
    geo_centers = []
    for mask in masks:
        if mask.sum() == 0:
            if "hull" in methods:
                hull_centers.append(np.zeros((1, 3)))
            if "geo" in methods:
                geo_centers.append(np.zeros((1, 3)))
            continue
        atom_positions = []
        for i in range(len(mask)):
            if mask[i] == 1:
                atom_positions.append(protein[i].node_position)
        atom_positions = np.concatenate(atom_positions, axis=0)
        if "hull" in methods:
            if len(atom_positions) < 4:
                hull_centers.append(np.mean(atom_positions, axis=0, keepdims=True))
            else:
                hull = ConvexHull(atom_positions)
                hull_centers.append(hull_center(hull)[None, :])
        if "geo" in methods:
            geo_centers.append(np.mean(atom_positions, axis=0, keepdims=True))
    ret = {}
    if "hull" in methods:
        hull_centers = np.concatenate(hull_centers, axis=0)
        ret["hull_centers"] = hull_centers
    if "geo" in methods:
        geo_centers = np.concatenate(geo_centers, axis=0)
        ret["geo_centers"] = geo_centers
    return ret


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the pocket mask error in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, num_res] with the predicted pocket masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_pockets] (where num_target_pockets is the number of ground-truth
                           pockets in the protein) containing the class labels
                 "pocket_masks": Tensor of dim [num_target_pockets, num_res] containing the target pocket masks
                 "res_mask": Tensor of dim [num_res] containing the mask for each residue

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_pockets)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        out_probs = outputs["pred_logits"].softmax(-1)  # [batch_size, num_queries, num_classes]
        out_masks = outputs["pred_masks"]  # [batch_size, num_queries, num_res]

        ret = []
        for i, target_per_protein in enumerate(targets):
            tgt_ids = target_per_protein["labels"]
            tgt_masks = target_per_protein["pocket_masks"]  # [num_target_pockets, num_res]
            tgt_res_mask = target_per_protein["res_mask"]

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_probs[i, :, tgt_ids] # [num_queries, num_target_pockets]

            # align the length of prediciton and target
            length = len(tgt_res_mask)
            pred_masks = out_masks[i][:, : length]  # [num_queries, num_res]

            # apply the residue mask
            tgt_masks = tgt_masks[:, tgt_res_mask]
            pred_masks = pred_masks[:, tgt_res_mask]

            cost_mask = batch_sigmoid_ce_loss_jit(pred_masks, tgt_masks)  # [num_queries, num_target_pockets]
            cost_dice = batch_dice_loss_jit(pred_masks, tgt_masks)  # [num_queries, num_target_pockets]

            # Final cost matrix
            C = self.cost_class * cost_class + self.cost_mask * cost_mask + self.cost_dice * cost_dice
            C = C.cpu()

            row_ids, col_ids = linear_sum_assignment(C)
            ret.append((torch.as_tensor(row_ids, dtype=torch.int64), torch.as_tensor(col_ids, dtype=torch.int64)))

        return ret


class SetCriterion(nn.Module):
    """ 
    This class computes the loss for Pocket Detection.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth pockets and predicted pockets
        2) we supervise the model based on this assignment
    """
    def __init__(self, model_conf):
        super().__init__()
        self.matcher = HungarianMatcher(
            cost_class=model_conf.matcher.cost_class, 
            cost_mask=model_conf.matcher.cost_mask,
            cost_dice=model_conf.matcher.cost_dice
        )
        self.criterion_class = nn.CrossEntropyLoss()
        self.model_conf = model_conf
        self.num_classes = model_conf.num_classes

        self.eos_coef = 0.1 #eos_coef
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.empty_weight = empty_weight
    
    def loss_labels(self, outputs, targets, indices):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_class = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight.to(src_logits.device))

        return loss_class, target_classes
    
    def loss_masks(self, outputs, targets, indices, num_masks):
        """only calculate the loss for the matched pocket masks"""
        loss_mask = 0
        loss_dice = 0
        for out_masks, t, ind in zip(outputs["pred_masks"], targets, indices):
            target_masks = t["pocket_masks"]    # [num_target_pockets, num_res]
            res_mask = t["res_mask"]
            row_ids, col_ids = ind
            out_masks = out_masks[row_ids]
            out_masks = out_masks[: , : len(res_mask)]   # align the length of prediciton and target
            out_masks = out_masks[:, res_mask]
            target_masks = target_masks[col_ids]
            target_masks = target_masks[:, res_mask]
            loss_mask += sigmoid_ce_loss_jit(out_masks, target_masks, num_masks)
            loss_dice += dice_loss_jit(out_masks, target_masks, num_masks)
        return loss_mask, loss_dice
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)

        # Compute the average number of target masks accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor([num_masks], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
            num_masks = torch.clamp(num_masks / dist.get_world_size(), min=1).item()
        else:
            num_masks = num_masks.item()

        loss_class, target_classes = self.loss_labels(outputs, targets, indices)
        loss_mask, loss_dice = self.loss_masks(outputs, targets, indices, num_masks)
        loss_dict = {"loss_cls": loss_class, "loss_mask": loss_mask, "loss_dice": loss_dice}
        loss_dict["target_classes"] = target_classes
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                loss_class_i, _ = self.loss_labels(aux_outputs, targets, indices)
                loss_mask_i, loss_dice_i = self.loss_masks(aux_outputs, targets, indices, num_masks)
                loss_dict.update({
                    f"loss_cls_{i}": loss_class_i,
                    f"loss_mask_{i}": loss_mask_i,
                    f"loss_dice_{i}": loss_dice_i,
                })
        return loss_dict

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
 
    def __init__(self, cost_logits: float = 1, cost_kpt: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box kptinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_logits = cost_logits
        self.cost_kpt = cost_kpt
        # self.cost_id = cost_id
        assert cost_logits != 0 or cost_kpt != 0, "all costs cant be 0"
 
    @torch.no_grad()
    def forward(self, outputs, targets, frame):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box kptinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box kptinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        num_frame, bs, num_queries = outputs["pred_logits"].shape[:3]
        out_logits = outputs["pred_logits"][frame].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_logiclasses]
        # out_id = outputs["pred_id"][frame].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_idclasses]
        out_kpt = outputs["pred_kpt"][frame].flatten(0, 1)  # [batch_size * num_queries, 14*3]
        

        tgt_cls = torch.cat([v["kpt_cls"][frame] for v in targets]).squeeze()
        tgt_kpt = torch.cat([v["kpt"][frame] for v in targets])
        cost_logits = -out_logits[:, tgt_cls]

        cost_kpt = torch.cdist(out_kpt, tgt_kpt, p=1)

        
        C = self.cost_kpt * cost_kpt + self.cost_logits * cost_logits
        # C = self.cost_kpt * cost_kpt
        C = C.view(bs, num_queries, -1).cpu()
        # size [20,14*3]
        sizes = [len(v["kpt"][frame]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        now = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        return now


def build_matcher(args):
    return HungarianMatcher(cost_logits=args.set_cost_class, cost_kpt=args.set_cost_kpt)


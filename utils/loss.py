import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.misc import is_dist_avail_and_initialized, get_world_size, accuracy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_logitsclass, matcher, weight_dict, logits_eos_coef, losses, num_classes=8):
        """ Create the criterion.
        Parameters:
            num_logitsclass: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_logitsclass = num_logitsclass
        # self.num_idclass = num_idclass
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.logits_eos_coef = logits_eos_coef
        # self.id_eos_coef = id_eos_coef
        self.losses = losses
        empty_weight_logits = torch.ones(self.num_logitsclass + 1)
        empty_weight_logits[0] = self.logits_eos_coef
        self.correct_num = 0
        # self.mpjpe = 0.0
        self.mpjpe_list = []
        self.mpjdle_list = []
        self.mpjdle_h_list = []
        self.mpjdle_v_list = []
        self.mpjdle_d_list = []

        self.mpjpe_thumb_list = []
        self.mpjdle_h_thumb_list = []
        self.mpjdle_v_thumb_list = []
        self.mpjdle_d_thumb_list = []

        self.mpjpe_index_list = []
        self.mpjdle_h_index_list = []
        self.mpjdle_v_index_list = []
        self.mpjdle_d_index_list = []

        self.mpjpe_middle_list = []
        self.mpjdle_h_middle_list = []
        self.mpjdle_v_middle_list = []
        self.mpjdle_d_middle_list = []

        self.mpjpe_ring_list = []
        self.mpjdle_h_ring_list = []
        self.mpjdle_v_ring_list = []
        self.mpjdle_d_ring_list = []

        self.mpjpe_pinky_list = []
        self.mpjdle_h_pinky_list = []
        self.mpjdle_v_pinky_list = []
        self.mpjdle_d_pinky_list = []
        
        self.num_classes = num_classes
        # self.kpt = torch.Tensor()
        # empty_weight_id = torch.ones(self.num_idclass + 1)
        # empty_weight_id[0] = self.id_eos_coef
        self.conf_matrix = [[0 for _ in range(8)] for _ in range(8)]
        self.register_buffer('empty_weight_logits', empty_weight_logits)
        # self.register_buffer('empty_weight_id', empty_weight_id)

    def loss_cls(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        target_cls_list = []
        target_cls_list_o = []
        for frame in range(0, 30):
            idx = self._get_src_permutation_idx(indices[frame])
            target_classes_o = torch.cat(
                [t["kpt_cls"][frame, J] for t, (_, J) in zip(targets, indices[frame])]).squeeze()
            target_classes = torch.full(src_logits[frame].shape[:2], 0,
                                        dtype=torch.int64, device=src_logits.device)
            target_classes[idx] = target_classes_o
            target_cls_list.append(target_classes)
            target_cls_list_o.append(target_classes_o)

        target_classes_af = torch.cat(target_cls_list, 0)
        target_classes_af_o = torch.cat(target_cls_list_o, 0)
        loss_ce = F.cross_entropy(src_logits.flatten(0, 2), target_classes_af.flatten(0, 1))
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            src_classes_list = []
            for frame in range(0, 30):
                idx = self._get_src_permutation_idx(indices[frame])
                src_logits_f = src_logits[frame]
                src_classes = src_logits_f[idx]
                src_classes_list.append(src_classes)
            src_class = torch.cat(src_classes_list, 0)

            losses['class_error'] = 100 - accuracy(src_class, target_classes_af_o)[0]
        return losses

    def loss_kpt(self, outputs, targets, indices, num_kpt):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_kpt' in outputs
        target_kpt_list_p = []
        target_kpt_list_o = []
        predict0 = []
        predict1 = []
        for frame in range(0, 30):
            idx = self._get_src_permutation_idx(indices[frame])

            target_kpt_o = torch.cat([t["kpt"][frame, J] for t, (_, J) in zip(targets, indices[frame])])
            src_kpt_f = outputs['pred_kpt'][frame][idx]
            target_kpt_list_p.append(src_kpt_f)
            target_kpt_list_o.append(target_kpt_o)

        #     idx0 = (torch.nonzero(idx[0] == 0).squeeze(), idx[1][torch.nonzero(idx[0] == 0)].squeeze())
        #     idx1 = (torch.nonzero(idx[0] == 1).squeeze(), idx[1][torch.nonzero(idx[0] == 1)].squeeze())
        #
        #     src_kpt0 = outputs['pred_kpt'][frame][0][idx0[1]]
        #     src_kpt1 = outputs['pred_kpt'][frame][1][idx1[1]]
        #     predict0.append(src_kpt0)
        #     predict1.append(src_kpt1)
        #
        #     # print()
        #
        # predict_kpt0 = torch.stack(predict0)
        # predict_kpt1 = torch.stack(predict1)
        # self.save_as_npy(predict_kpt0, predict_kpt1, targets)

        src_kpt = torch.cat(target_kpt_list_p, 0)
        target_kpt = torch.cat(target_kpt_list_o, 0)

        loss_kpt = F.mse_loss(src_kpt, target_kpt, reduction='none')
        result = self.calc_mpjpe(src_kpt, target_kpt)

        self.mpjpe = self.mpjpe_list.append(result[0])
        self.mpjdle = self.mpjdle_list.append(result[1])
        self.mpjdle_h = self.mpjdle_h_list.append(result[2])
        self.mpjdle_v = self.mpjdle_v_list.append(result[3])
        self.mpjdle_d = self.mpjdle_d_list.append(result[4])

        self.mpjpe_thumb = self.mpjpe_thumb_list.append(result[5])
        self.mpjdle_h_thumb = self.mpjdle_h_thumb_list.append(result[6])
        self.mpjdle_v_thumb = self.mpjdle_v_thumb_list.append(result[7])
        self.mpjdle_d_thumb = self.mpjdle_d_thumb_list.append(result[8])

        self.mpjpe_index = self.mpjpe_index_list.append(result[9])
        self.mpjdle_h_index = self.mpjdle_h_index_list.append(result[10])
        self.mpjdle_v_index = self.mpjdle_v_index_list.append(result[11])
        self.mpjdle_d_index = self.mpjdle_d_index_list.append(result[12])

        self.mpjpe_middle = self.mpjpe_middle_list.append(result[13])
        self.mpjdle_h_middle = self.mpjdle_h_middle_list.append(result[14])
        self.mpjdle_v_middle = self.mpjdle_v_middle_list.append(result[15])
        self.mpjdle_d_middle = self.mpjdle_d_middle_list.append(result[16])

        self.mpjpe_ring = self.mpjpe_ring_list.append(result[17])
        self.mpjdle_h_ring = self.mpjdle_h_ring_list.append(result[18])
        self.mpjdle_v_ring = self.mpjdle_v_ring_list.append(result[19])
        self.mpjdle_d_ring = self.mpjdle_d_ring_list.append(result[20])

        self.mpjpe_pinky = self.mpjpe_pinky_list.append(result[21])
        self.mpjdle_h_pinky = self.mpjdle_h_pinky_list.append(result[22])
        self.mpjdle_v_pinky = self.mpjdle_v_pinky_list.append(result[23])
        self.mpjdle_d_pinky = self.mpjdle_d_pinky_list.append(result[24])

        losses = {}
        losses['loss_kpt'] = loss_kpt.sum() / num_kpt
        return losses

    def save_as_npy(self, predict0, predict1, targets):
        if int(targets[0]['label']) not in [6, 7]:
            predict0 = predict0.reshape((-1, 21, 3))
        else:
            predict0 = predict0.reshape((30, 2, 21, 3)).reshape((30, 42, 3))
        if int(targets[1]['label']) not in [6, 7]:
            predict1 = predict1.reshape((-1, 21, 3))
        else:
            predict1 = predict1.reshape((30, 2, 21, 3)).reshape((30, 42, 3))

        np.save('/home/lvyizhe/data/kpt_loss_12/' + '_'.join(
            map(lambda x: f'{x:02d}', targets[0]['filename'].cpu().numpy())) + '.npy', predict0.cpu().numpy())
        np.save('/home/lvyizhe/data/kpt_loss_12/' + '_'.join(
            map(lambda x: f'{x:02d}', targets[1]['filename'].cpu().numpy())) + '.npy', predict1.cpu().numpy())

    def loss_label(self, outputs, targets, indices, num_kpt, log=True):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        # assert 'pred_label' in outputs

        assert 'pred_label' in outputs
        src_labels = outputs['pred_label']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_labels.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_labels.device)
        target_classes[idx] = target_classes_o
        print(src_labels.shape, '     ', src_labels.transpose(1, 2).shape)
        loss_label = F.cross_entropy(src_labels.transpose(1, 2), target_classes)
        losses = {'loss_label': loss_label}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['label_error'] = 100 - accuracy(src_labels[idx], target_classes_o)[0]
        return losses

        # print(len(targets))
        target_label = [int(v['label']) for v in targets]
        target_label = torch.tensor(target_label).cuda()
        pred_label = outputs['pred_cls'].data.max(1)[1]
        losses = {}
        loss_label = F.cross_entropy(outputs['pred_cls'], target_label)
        losses['loss_label'] = loss_label
        # print('correct_num: ', int(pred_label.eq(target_label).sum()))
        self.conf_matrix = self.get_conf_matrix(pred_label, target_label, self.conf_matrix)
        self.correct_num += pred_label.eq(target_label).sum()
        # print('cur_num: ', int(self.correct_num))
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def calc_mpjpe(self, output_kpt, target_kpt):
        calc_joint = True
        result = []
        new_pred = output_kpt.flatten(0, 1).reshape(-1, 21, 3)
        real = target_kpt.flatten(0, 1).reshape(-1, 21, 3)
        pjpe = torch.sqrt(torch.pow(real - new_pred, 2).sum(-1)) * 1000
        pjdle = torch.abs(real - new_pred) * 1000
        pjdle_h = torch.abs(real[:,:,0] - new_pred[:,:,0]) * 1000
        pjdle_v = torch.abs(real[:,:,1] - new_pred[:,:,1]) * 1000
        pjdle_d = torch.abs(real[:,:,2] - new_pred[:,:,2]) * 1000
        result.append(pjpe.mean().cpu())
        result.append(pjdle.mean().cpu())
        result.append(pjdle_h.mean().cpu())
        result.append(pjdle_v.mean().cpu())
        result.append(pjdle_d.mean().cpu())
        if calc_joint:
            pjpe_thumb = torch.sqrt(torch.pow(real[:,1:5,:] - new_pred[:,1:5,:], 2).sum(-1)) * 1000
            pjdle_h_thumb = torch.abs(real[:,1:5,0] - new_pred[:,1:5,0]) * 1000
            pjdle_v_thumb = torch.abs(real[:,1:5,1] - new_pred[:,1:5,1]) * 1000
            pjdle_d_thumb = torch.abs(real[:,1:5,2] - new_pred[:,1:5,2]) * 1000
            result.append(pjpe_thumb.mean().cpu())
            result.append(pjdle_h_thumb.mean().cpu())
            result.append(pjdle_v_thumb.mean().cpu())
            result.append(pjdle_d_thumb.mean().cpu())

            pjpe_index = torch.sqrt(torch.pow(real[:,5:9,:] - new_pred[:,5:9,:], 2).sum(-1)) * 1000
            pjdle_h_index = torch.abs(real[:,5:9,0] - new_pred[:,5:9,0]) * 1000
            pjdle_v_index = torch.abs(real[:,5:9,1] - new_pred[:,5:9,1]) * 1000
            pjdle_d_index = torch.abs(real[:,5:9,2] - new_pred[:,5:9,2]) * 1000
            result.append(pjpe_index.mean().cpu())
            result.append(pjdle_h_index.mean().cpu())
            result.append(pjdle_v_index.mean().cpu())
            result.append(pjdle_d_index.mean().cpu())

            pjpe_middle = torch.sqrt(torch.pow(real[:,9:13,:] - new_pred[:,9:13,:], 2).sum(-1)) * 1000
            pjdle_h_middle = torch.abs(real[:,9:13,0] - new_pred[:,9:13,0]) * 1000
            pjdle_v_middle = torch.abs(real[:,9:13,1] - new_pred[:,9:13,1]) * 1000
            pjdle_d_middle = torch.abs(real[:,9:13,2] - new_pred[:,9:13,2]) * 1000
            result.append(pjpe_middle.mean().cpu())
            result.append(pjdle_h_middle.mean().cpu())
            result.append(pjdle_v_middle.mean().cpu())
            result.append(pjdle_d_middle.mean().cpu())

            pjpe_ring = torch.sqrt(torch.pow(real[:,13:17,:] - new_pred[:,13:17,:], 2).sum(-1)) * 1000
            pjdle_h_ring = torch.abs(real[:,13:17,0] - new_pred[:,13:17,0]) * 1000
            pjdle_v_ring = torch.abs(real[:,13:17,1] - new_pred[:,13:17,1]) * 1000
            pjdle_d_ring = torch.abs(real[:,13:17,2] - new_pred[:,13:17,2]) * 1000
            result.append(pjpe_ring.mean().cpu())
            result.append(pjdle_h_ring.mean().cpu())
            result.append(pjdle_v_ring.mean().cpu())
            result.append(pjdle_d_ring.mean().cpu())

            pjpe_pinky = torch.sqrt(torch.pow(real[:,17:21,:] - new_pred[:,17:21,:], 2).sum(-1)) * 1000
            pjdle_h_pinky = torch.abs(real[:,17:21,0] - new_pred[:,17:21,0]) * 1000
            pjdle_v_pinky = torch.abs(real[:,17:21,1] - new_pred[:,17:21,1]) * 1000
            pjdle_d_pinky = torch.abs(real[:,17:21,2] - new_pred[:,17:21,2]) * 1000
            result.append(pjpe_pinky.mean().cpu())
            result.append(pjdle_h_pinky.mean().cpu())
            result.append(pjdle_v_pinky.mean().cpu())
            result.append(pjdle_d_pinky.mean().cpu())

        return result

    def get_conf_matrix(self, pred, truth, conf_matrix):
        p = pred.tolist()
        l = truth.tolist()
        for i in range(len(p)):
            conf_matrix[l[i]][p[i]] += 1
        return conf_matrix

    def write_to_file(self, conf_matrix, path):
        conf_matrix_m = conf_matrix
        for x in range(len(conf_matrix_m)):
            base = sum(conf_matrix_m[x])
            for y in range(len(conf_matrix_m[0])):
                conf_matrix_m[x][y] = format(conf_matrix_m[x][y] / base, '.2f')
        df = pd.DataFrame(conf_matrix_m)
        df.to_csv(path + '.csv')

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        '''
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        '''
        loss_map = {
            'cls': self.loss_cls,
            'kpt': self.loss_kpt,
            # 'label': self.loss_label
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices_list = []
        for i in range(0, 30):
            indices = self.matcher(outputs_without_aux, targets, i)
            indices_list.append(indices)
        num_boxes = 0
        for t in targets:
            f, n, _ = t["kpt"].size()
            num_boxes = num_boxes + f * n
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        '''if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)'''
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices_list, num_boxes))

        return losses

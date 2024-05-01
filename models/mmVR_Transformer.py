import torch
import torch.nn.functional as F
from torch import nn
from utils.matcher import build_matcher
from utils.loss import SetCriterion
from models.backbone import build_backbone
from models.position_encoding import build_position_encoding
from models.Transformer_layers import TransformerEncoderLayer, TransformerEncoder, \
    TransformerPoseDecoderLayer, TransformerPoseDecoder, \
    TransformerTemporalDecoderLayer, TransformerTemporalDecoder


class prediction_head(nn.Module):
    def __init__(self, hidden_dim, num_logitsclasses, num_idclasses):
        super().__init__()
        self.logitsclass_embed = nn.Linear(hidden_dim, num_logitsclasses + 1)

        self.kpt_embed = MLP(hidden_dim, hidden_dim, 21 * 3, 3)

    def forward(self, tgt):
        outputs_logi = self.logitsclass_embed(tgt)
        outputs_kpt = self.kpt_embed(tgt)
        return outputs_logi, outputs_kpt


class classification_head(nn.Module):
    def __init__(self, feature_dim=30 * 100 * 512, hidden_dim=512, num_classes=8):
        super().__init__()
        self.class_embed = nn.Linear(feature_dim, num_classes)
        self.class_embed2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.class_embed(x)
        output = self.class_embed2(x)
        return output


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class mmVR_Transformer(nn.Module):

    def __init__(self, args, backbone, backbone_name, d_model=512, nhead=8, num_frames=30, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, num_logitsclass=1,
                 num_queries=100, return_intermediate_dec=False, num_class=8):
        super().__init__()
        self.hidden_dim = d_model
        self.mmwave_backbone = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            nn.Conv3d(64, 64, kernel_size=(3, 2, 2), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),


        )
        self.fc_mmwave = nn.Linear(64 * 8 * 4, 512)
        self.imu_backbone = nn.LSTM(input_size=6, hidden_size=512, num_layers=2, batch_first=True)
        self.position_embedding = build_position_encoding(args)
        self.num_channels = 512 if backbone_name in ('resnet18', 'resnet34') else 2048

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        encoder_layer_imu = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm_imu = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder_imu = TransformerEncoder(encoder_layer_imu, num_encoder_layers, encoder_norm_imu)

        posedecoder_layer = TransformerPoseDecoderLayer(d_model, nhead, dim_feedforward,
                                                        dropout, activation, normalize_before).cuda()
        posedecoder_norm = nn.LayerNorm(d_model).cuda()
        self.posedecoder = TransformerPoseDecoder(posedecoder_layer, num_decoder_layers, posedecoder_norm,
                                                  return_intermediate=return_intermediate_dec).cuda()

        temporaldecoder_layer = TransformerTemporalDecoderLayer(d_model, nhead, dim_feedforward,
                                                                dropout, activation, normalize_before).cuda()
        temporaldecoder_norm = nn.LayerNorm(d_model).cuda()
        self.temporaldecoder = TransformerTemporalDecoder(temporaldecoder_layer, num_frames, temporaldecoder_norm).cuda()
        # self.pos_embed = nn.Parameter(torch.zeros((30, 32, self.hidden_dim)))
        self.posequery_embed = nn.Embedding(30, self.hidden_dim)
        self.temporalquery_embed = nn.Embedding(num_queries, self.hidden_dim)

        self.prediction_head = prediction_head(self.hidden_dim, num_logitsclass, num_class)
        # self.classification_head = classification_head(num_frames*512*100, d_model, num_class)
        self.input_proj = nn.Conv2d(self.num_channels, self.hidden_dim, kernel_size=1)
        self.feature_linear = nn.Linear(256 * 128, self.hidden_dim)
        self.feature_linear_imu = nn.Linear(6, self.hidden_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, imu):

        src = src.unsqueeze(1)
        bs = src.shape[0]

        src = self.mmwave_backbone(src)

        src = src.permute(0, 2, 1, 3, 4).contiguous()  # batch, time, channels, height, width

        src = src.view(src.shape[0], src.shape[1], -1)  # flatten
        src = self.fc_mmwave(src)
        imu, _ = self.imu_backbone(imu)
        pos_embed_mmwave = self.position_embedding(src.reshape(bs, 30, 32, 16))
        pos_embed_mmwave = pos_embed_mmwave.flatten(2).permute(1, 0, 2)
        pos_embed_imu = self.position_embedding(imu.reshape(bs, 30, 32, 16))
        pos_embed_imu = pos_embed_imu.flatten(2).permute(1, 0, 2)

        pos_embed = 0.9 * pos_embed_mmwave + 0.1 * pos_embed_imu


        src = src.permute(1, 0, 2)
        imu = imu.permute(1, 0, 2)

        memory_imu = self.encoder_imu(imu, pos=pos_embed_imu)

        posequery_embed = self.posequery_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        temporalquery_embed = self.temporalquery_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        tgt_pose = torch.zeros_like(posequery_embed)
        tgt_tem = torch.zeros_like(temporalquery_embed)

        memory = self.encoder(src, pos=pos_embed_mmwave)

        memory = 0.1 * memory_imu + 0.9 * memory

        hs = self.posedecoder(tgt_pose, memory, pos=pos_embed, query_pos=posequery_embed)

        pose_memory = hs[-1, :, :, :]
        temporal_out = self.temporaldecoder(tgt_tem, pose_memory, memory,
                                            pos=pos_embed, query_pos=temporalquery_embed)
        temporal_out = temporal_out.transpose(1, 2)
        outputs_logi, outputs_kpt = self.prediction_head(temporal_out)
        out = {'pred_logits': outputs_logi, 'pred_kpt': outputs_kpt}
        return out


def build_model(args):
    device = torch.device(args.device)
    backbone = build_backbone(args)
    model = mmVR_Transformer(args, backbone, args.backbone_name, d_model=512, nhead=8, num_frames=30, num_encoder_layers=6,
                          num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                          activation="relu", normalize_before=True, num_logitsclass=1,
                          num_queries=100, return_intermediate_dec=True, num_class=8
                          )

    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_kpt': args.kpt_loss_coef}
    losses = ['cls', 'kpt']
    criterion = SetCriterion(num_logitsclass=1, matcher=matcher, weight_dict=weight_dict,
                             logits_eos_coef=args.cls_loss_coef, losses=losses)
    criterion.to(device)
    return model, criterion

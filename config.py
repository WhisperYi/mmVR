import argparse
import math
parser = argparse.ArgumentParser(description='Hyper-parameters management')

parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
parser.add_argument('--lr', default=1e-4, type=float)  # 1e-4
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--val_batch_size', default=1, type=int)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--epoch', default=200, type=int)
parser.add_argument('--lr_drop', default=200, type=int)
parser.add_argument('--clip_max_norm', default=0.1, type=float,
                    help='gradient clipping max norm')
parser.add_argument('--mode', default=True)
parser.add_argument('--backbone_name', type=str, default='resnet18')
parser.add_argument('--backbone_pretrain', type=bool, default=True)
# Model parameters
parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
# * Backbone
parser.add_argument('--dilation', action='store_true',
                    help="If true, we replace stride with dilation in the last convolutional block (DC5)")
parser.add_argument('--position_embedding', default='learned', type=str, choices=('sine', 'learned'),
                    help="Type of positional embedding to use on top of the image features")

# * Transformer
parser.add_argument('--enc_layers', default=6, type=int,
                    help="Number of encoding layers in the transformer")
parser.add_argument('--dec_layers', default=6, type=int,
                    help="Number of decoding layers in the transformer")
parser.add_argument('--dim_feedforward', default=2048, type=int,
                    help="Intermediate size of the feedforward layers in the transformer blocks")
parser.add_argument('--hidden_dim', default=256, type=int,
                    help="Size of the embeddings (dimension of the transformer)")
parser.add_argument('--dropout', default=0.1, type=float,
                    help="Dropout applied in the transformer")
parser.add_argument('--nheads', default=8, type=int,
                    help="Number of attention heads inside the transformer's attentions")
parser.add_argument('--num_queries', default=100, type=int,
                    help="Number of query slots")
parser.add_argument('--pre_norm', action='store_true')

# * Segmentation
parser.add_argument('--masks', action='store_true',
                    help="Train segmentation head if the flag is provided")

# Loss
parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                    help="Disables auxiliary decoding losses (loss at each layer)")
# * Matcher
parser.add_argument('--set_cost_class', default=5, type=float,
                    help="Class coefficient in the matching cost")
parser.add_argument('--set_cost_kpt', default=25, type=float,
                    help="L1 box coefficient in the matching cost")
parser.add_argument('--set_cost_id', default=2, type=float,
                    help="giou box coefficient in the matching cost")
# * Loss coefficients
parser.add_argument('--cls_loss_coef', default=1, type=float)
parser.add_argument('--kpt_loss_coef', default=5, type=float)
parser.add_argument('--id_loss_coef', default=2, type=float)


# dataset parameters
parser.add_argument('--dataset_root', default='./data')
parser.add_argument('--dataset_root_kpt', default='./data')
parser.add_argument('--remove_difficult', action='store_true')

parser.add_argument('--output_dir', default='',
                    help='path where to save, empty for no saving')
parser.add_argument('--device', default='cuda',
                    help='device to use for training / testing')
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--resume', default='', help='resume from checkpoint')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='start epoch')
parser.add_argument('--eval', action='store_true', default=False)
parser.add_argument('--num_workers', default=2, type=int)

args = parser.parse_args()



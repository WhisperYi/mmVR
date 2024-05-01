import os
import logging
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

log = logging.getLogger(__name__)

class PN_Dataset(Dataset):
    def __init__(self, dataset_root, mode):
        super(PN_Dataset, self).__init__()
        self.data_root = dataset_root
        self.is_train = mode
        if self.is_train:
            self.file = os.path.join(self.data_root, 'train_list.txt')
        else:
            self.file = os.path.join(self.data_root, 'eval_list.txt')
        self.path = self.data_root
        file = open(self.file)
        data_list = file.readlines()
        self.data = {
            'file_name': list(),
            'people_id': list(),
            'label': list(),
            'vector': list()
        }
        for string in data_list:
            label = int(string.split(',')[3]) - 1
            # else:
            self.data['label'].append(label)
            self.data['file_name'].append(string.split(',')[0])
            self.data['people_id'].append(string.split(',')[1])

    def __getitem__(self, index):
        data_name = self.data['file_name'][index]
        label = self.data['label'][index]
        people_id = self.data['people_id'][index]
        keypoint_data = self.load_keypoint(data_name)
        if label in [0, 1, 2, 3, 4, 5]:
            padding = torch.zeros(30, 21, 3)
            if people_id in ['02', '04', '05', '12']:
                keypoint_data = torch.cat((keypoint_data, padding), dim=1)
            else:
                keypoint_data = torch.cat((padding, keypoint_data), dim=1)
        return keypoint_data, label

    def __len__(self):
        return len(self.data['file_name'])

    def load_keypoint(self, filename):
        keypoint_data = np.load(os.path.join(self.path, 'keypoint_output', filename + '.npy'))
        return torch.from_numpy(keypoint_data).float()


def build_dataset(dataset_root, mode):
    return PN_Dataset(dataset_root, mode)

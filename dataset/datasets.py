import os
import logging
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import h5py

log = logging.getLogger(__name__)

class HM_Dataset(Dataset):
    def __init__(self, dataset_root, mode):
        super(HM_Dataset, self).__init__()
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
            'person_id': list(),
            'scene': list(),
            'label': list()
        }
        for string in data_list:
            label = int(string.split(',')[3]) - 1
            self.data['file_name'].append(string.split(',')[0])
            self.data['person_id'].append(string.split(',')[1])
            self.data['scene'].append(string.split(',')[2])
            self.data['label'].append(label)


        log.info("加载完毕HM数据集")

    def __getitem__(self, index):
        data_name = self.data['file_name'][index]
        mmwave_data = self.load_mmwave(data_name)
        imu_data = self.load_imu(data_name)
        keypoint_data = self.load_keypoint(data_name)
        targets = self.get_item_single_frame(keypoint_data, index)
        return mmwave_data, imu_data, targets

    def __len__(self):
        return len(self.data['file_name'])

    def load_mmwave(self, filename):
        mmWave_data = h5py.File(os.path.join(self.path, 'mmwave', filename + '.mat'), "r").get("data")
        mmWave_data = np.array(mmWave_data).transpose((2, 1, 0))
        indices_to_keep = np.linspace(0, 39, 30, dtype=int)
        mmWave_data = mmWave_data[indices_to_keep]
        return torch.from_numpy(mmWave_data).float()

    def load_imu(self, filename):
        imu_data = np.load(os.path.join(self.path, 'imu', filename + '.npy'))
        indices_to_keep = np.linspace(0, 39, 30, dtype=int)
        imu_data = imu_data[indices_to_keep]
        return torch.from_numpy(imu_data).float()

    def load_keypoint(self, filename):
        keypoint_data = np.load(os.path.join(self.path, 'keypoint_gt', filename + '.npy'))
        if keypoint_data.shape[1] == 21:
            keypoint_data = keypoint_data.reshape(30, 1, -1)
        else:
            keypoint_data = keypoint_data.reshape(30, 2, -1)
        return torch.from_numpy(keypoint_data).float()

    def get_item_single_frame(self, keypoint, index):
        NumOfHand = keypoint.shape[1]
        kpt_cls = self.get_cls(NumOfHand)
        kpt_cls = torch.LongTensor(kpt_cls).unsqueeze(0).repeat(30, 1, 1)

        numbers = [int(num) for num in self.data['file_name'][index].split('_')]
        target = {'kpt': keypoint, 'kpt_cls': kpt_cls,
                  'label': torch.as_tensor(self.data['label'][index]),
                  'filename': torch.tensor(numbers)
                }
        return target

    def get_cls(self, numofperson):
        cls_tensor = np.zeros((numofperson, 1))
        cls_tensor[:, 0] = 1
        return cls_tensor

def build_dataset(dataset_root, mode):
    return HM_Dataset(dataset_root, mode)
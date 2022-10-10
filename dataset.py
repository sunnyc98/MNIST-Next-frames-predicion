import random
import numpy as np
import torch

class MovingMNIST(torch.utils.data.Dataset):
    def __init__(self, fdir, split, previous_num, prediction_num):
        assert split in ['train', 'test']
        self.split = split
        self.fdir = fdir
        self.previous_num = previous_num
        self.prediction_num = prediction_num
        self.dataset = self.load_MovingMNIST()

    def load_MovingMNIST(self):
        data = np.load(self.fdir)
        data = np.transpose(data, [1,2,3,0]) / (255/2) -1

        test_index = range(0,3)
        if self.split == 'train':
            index = range(0,10000)
            index = [x for x in index if x not in test_index]
            data = data[index]
        else:
            data = data[test_index]
        return data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        imgs = self.dataset[index]

        if self.split == 'train':
            seq_idx = random.randint(0, 20-(self.previous_num + self.prediction_num))
            imgs = imgs[:,:, seq_idx:seq_idx + self.previous_num + self.prediction_num]
        else:
            imgs = imgs[:,:,:self.previous_num + self.prediction_num]
        imgs_previous_seq = imgs[:,:,:self.previous_num]
        imgs_gt_next_frame = imgs[:,:, self.previous_num:self.previous_num+self.prediction_num]

        imgs_previous_seq = torch.FloatTensor(imgs_previous_seq)
        imgs_gt_next_frame = torch.FloatTensor(imgs_gt_next_frame)

        return imgs_previous_seq, imgs_gt_next_frame

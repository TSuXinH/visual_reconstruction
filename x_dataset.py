import os
import cv2
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset


stim_start_point = [134, 1515, 2896, 4278, 5659, 7040]
stim_end_point = [1138, 2519, 3900, 5282, 6663, 8044]
stim_point_arr = np.array([stim_start_point, stim_end_point]).T


class CustomDataset(Dataset):
    """
    Construct a custom dataset combine neural response and corresponding video frame.
    The length is number of the neural stimulus points.
    video shape: 256 * 256 temporarily
    neu_trans: do basic transformations to neural responses
    video_trans: do basic transformations to video frames
    todo: the most important question: align the stimuli and the responses, temporarily finished in function `cal_video_idx`
    """

    def __init__(self, neu_data, video_path, period_len, neu_trans, video_trans, mode='train'):
        super().__init__()
        # neural data should be reshaped as [stimulus time, neuron number]
        self.neu_data = neu_data.T
        self.video_path = video_path
        self.neu_trans = neu_trans
        self.video_trans = video_trans
        self.period_len = period_len
        self.video_len = len(os.listdir(self.video_path))
        self.mode = mode
        if self.mode.lower() == 'train' or self.mode.lower() == 'valid':
            _len = len(self.neu_data) // self.period_len * (self.period_len - 1)
            self.data = self.neu_data[: _len]
        elif self.mode.lower() == 'test':
            _len = len(self.neu_data) // self.period_len
            self.data = self.neu_data[-_len:]
        else:
            raise NotImplementedError

    def __repr__(self):
        return 'custom dataset\ndata shape: {}\nvideo length: {}'.format(self.data.shape, self.video_len)

    def __getitem__(self, item):
        data = self.data[item]
        idx = cal_video_idx(item % self.period_len)
        tmp_dir = '{}.jpg'.format(idx)
        target = Image.open(os.path.join(self.video_path, tmp_dir))
        return self.neu_trans(data), self.video_trans(target)

    def __len__(self):
        return len(self.data)


def cal_video_idx(neu_idx):
    """
    Calculate the video index in the storing file through a certain neural time point index.
    neural sampling period: 13 / 30 sec
    video sampling period: 1 / 25 sec
    """
    return int(np.round(neu_idx * 13 / 30 * 25).item())


def proc_neu_data(neu_mat, stim_array):
    """
    Split the neural responses via stimulus time.
    """
    f_list = []
    for idx in range(len(stim_array)):
        f_list.append(neu_mat[:, stim_point_arr[idx, 0]: stim_point_arr[idx, 1]])
    f_tensor = np.concatenate(f_list, axis=-1)
    return f_tensor


import torch
from torch import nn
import os
from PIL import Image
from easydict import EasyDict
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, ToTensor
import matplotlib.pyplot as plt

""" This dictionary is used for auto encoder archeture"""
arch_params = {
    'ae_network_type': 'strides_only',
    'ae_padding_type': 'same',
    'ae_batch_norm': False,
    'ae_batch_norm_momentum': None,
    'symmetric_arch': 1,
    'ae_encoding_n_channels': [32, 64, 128, 256, 512],
    'ae_encoding_kernel_size': [5, 5, 5, 5, 5],
    'ae_encoding_stride_size': [2, 2, 2, 2, 5],
    'ae_encoding_layer_type': ['conv', 'conv', 'conv', 'conv', 'conv'],
    'ae_decoding_last_FF_layer': 0,
    'with_sigmoid': False,
    'n_input_channels': 1,
    'y_pixels': 256,
    'x_pixels': 256,
    'ae_input_dim': [1, 256, 256],
    'n_ae_latents': 128,
    'model_type': 'conv',
    'model_class': 'ae',
    'ae_encoding_x_dim': [128, 64, 32, 16, 4],
    'ae_encoding_y_dim': [128, 64, 32, 16, 4],
    'ae_encoding_x_padding': [[1, 2], [1, 2], [1, 2], [1, 2], [2, 2]],
    'ae_encoding_y_padding': [[1, 2], [1, 2], [1, 2], [1, 2], [2, 2]],
    'ae_decoding_x_dim': [16, 32, 64, 128, 256],
    'ae_decoding_y_dim': [16, 32, 64, 128, 256],
    'ae_decoding_x_padding': [[2, 2], [1, 2], [1, 2], [1, 2], [1, 2]],
    'ae_decoding_y_padding': [[2, 2], [1, 2], [1, 2], [1, 2], [1, 2]],
    'ae_decoding_n_channels': [256, 128, 64, 32, 1],
    'ae_decoding_kernel_size': [5, 5, 5, 5, 5],
    'ae_decoding_stride_size': [5, 2, 2, 2, 2],
    'ae_decoding_layer_type': ['convtranspose', 'convtranspose', 'convtranspose', 'convtranspose', 'convtranspose'],
    'ae_decoding_starting_dim': [512, 4, 4]
}


class Meter:
    def __init__(self):
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, item):
        return NotImplementedError

    def clear(self):
        self.avg = 0.
        self.sum = 0.
        self.count = 0


class LossMeter(Meter):
    def __init__(self):
        super().__init__()
        self.last = 0.

    def update(self, item):
        self.sum += item
        self.count += 1
        self.avg = self.count

    def save_last(self):
        self.last = self.avg

    def do_save_ckpt(self):
        return self.avg > self.last


class FrameDataset(Dataset):
    def __init__(self, frame_path, trans=None):
        super().__init__()
        self.frame_path = frame_path
        self.trans = trans

    def __getitem__(self, item):
        dir_list = os.listdir(self.frame_path)
        idx_name = dir_list[item]
        idx = os.path.join(self.frame_path, idx_name)
        img = Image.open(idx).convert('L')
        res = self.trans(img) if self.trans else img
        res = torch.FloatTensor(res)
        return res

    def __len__(self):
        return len(os.listdir(self.frame_path))


class Train:
    def __init__(self, model, loader, criterion, optimizer, args):
        super().__init__()
        self.loader = loader
        self.model = model.to(args.device)
        self.args = args
        self.criterion = criterion.to(args.device)
        self.optimizer = optimizer

    def train(self):
        last_loss = torch.inf
        for epoch in range(self.args.max_epoch):
            res = self.train_once()
            print('epoch: {}'.format(epoch))
            if 'loss' in res:
                print('loss: {:.4f}'.format(res.loss))
                if self.args.tensorboard:
                    self.args.tensorboard.add_scalar(tag='train loss', scalar_value=res.loss, global_step=epoch+1)
                if res.loss < last_loss:
                    state = {'model': self.model.state_dict(), 'epoch': epoch}
                    torch.save(state, self.args.ckpt_path)
                    last_loss = res.loss
        print()

    def train_once(self):
        if self.args.alter_train_method:
            return self.args.alter_train_method(self)
        if self.args.with_label:
            return self.train_with_label()
        else:
            return self.train_without_label()

    def train_with_label(self):
        raise NotImplementedError

    def train_without_label(self):
        self.model.train()
        loss_epoch = .0
        for idx, data in enumerate(self.loader):
            data = data.to(self.args.device)
            c, p, _ = data.shape[-3:]
            if self.args.ten_crop:
                data = data.reshape(-1, c, p, p)
            pred, _ = self.model(data)
            loss = self.criterion(pred, data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_epoch += loss.cpu().item()
            print('step: [{}]/[{}], loss: {:.6f}'.format(idx, len(self.loader), loss.cpu().item()))
        rec = EasyDict()
        rec.loss = loss_epoch / len(self.loader)
        return rec


def test_once(model, img, device):
    model.eval()
    model.to(device)
    img = torch.FloatTensor(ToTensor()(img)).unsqueeze(0).to(device)
    pred, _ = model(img)
    return pred.cpu()


def tensor2visual(mat):
    mat = torch.squeeze(mat)
    img = ToPILImage()(mat)
    img.show()


def test_show_contrast(model, img, device):
    model.eval()
    model.to(device)
    img_proc = torch.FloatTensor(ToTensor()(img)).unsqueeze(0).to(device)
    pred, _ = model(img_proc)
    pred = torch.squeeze(pred)
    res = torch.cat((torch.squeeze(img_proc).cpu(), pred.cpu()), dim=-1)
    res = ToPILImage()(res)
    res.show()

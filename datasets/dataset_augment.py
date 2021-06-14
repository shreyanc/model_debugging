import numpy as np
import torch
import torch.utils.data


def flip_ver(x):
    return x.flip(1)


def flip_hor(x):
    return x.flip(2)


class DsetSSFlipRand(torch.utils.data.Dataset):
    def __init__(self, dset):
        self.dset = dset

    def __getitem__(self, index):
        path, x = self.dset[index][:2]
        label = np.random.randint(2)
        if label == 0:
            x = x
        else:
            x = x.flip(1)
        return path, x, label

    def __len__(self):
        return len(self.dset)


class DsetSSMixup(torch.utils.data.Dataset):
    def __init__(self, dset):
        self.dset = dset

    def __getitem__(self, index):
        path, x = self.dset[index][:2]
        rand_idx = np.random.randint(len(self.dset))
        if rand_idx == index: rand_idx += 1
        path1, x1 = self.dset[rand_idx][:2]
        label = np.random.randint(2)

        if label == 0:
            x = x
        else:
            split_pt = x.shape[1] // 2  # + np.random.randint(-100,100)
            x = torch.cat([x[:, :split_pt], x1[:, split_pt:]], 1)
        return path, x, label

    def __len__(self):
        return len(self.dset)


class DsetSSTimeSwap(torch.utils.data.Dataset):
    def __init__(self, dset):
        self.dset = dset

    def __getitem__(self, index):
        path, x = self.dset[index][:2]
        label = np.random.randint(2)
        if label == 0:
            x = x
        else:
            lenx = x.shape[1]
            x = torch.cat([x[:, lenx // 2:], x[:, :lenx // 2]], 1)
        return path, x, label

    def __len__(self):
        return len(self.dset)


def tensor_rot_90(x):
    return x.flip(2).transpose(1, 2)


def tensor_rot_90_digit(x):
    return x.transpose(1, 2)


def tensor_rot_180(x):
    return x.flip(2).flip(1)


def tensor_rot_180_digit(x):
    return x.flip(2)


def tensor_rot_270(x):
    return x.transpose(1, 2).flip(2)


class DsetSSRotRand(torch.utils.data.Dataset):
    def __init__(self, dset, digit=True):
        self.dset = dset
        self.digit = digit

    def __getitem__(self, index):
        image = self.dset[index][0]
        label = np.random.randint(4)
        if label == 1:
            if self.digit:
                image = tensor_rot_90_digit(image)
            else:
                image = tensor_rot_90(image)
        elif label == 2:
            if self.digit:
                image = tensor_rot_180_digit(image)
            else:
                image = tensor_rot_180(image)
        elif label == 3:
            image = tensor_rot_270(image)
        return image, label

    def __len__(self):
        return len(self.dset)

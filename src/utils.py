import sys
import builtins
import torch
import math


class Printer(object):
    """Class for printing output by refreshing the same line in the console, e.g. for indicating progress of a process"""

    def __init__(self, console=True):

        if console:
            self.print = self.dyn_print
        else:
            self.print = builtins.print

    @staticmethod
    def dyn_print(data):
        """Print things to stdout on one line, refreshing it dynamically"""
        sys.stdout.write("\r\x1b[K" + data.__str__())
        sys.stdout.flush()

def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def readable_time(time_difference):
    """Convert a float measuring time difference in seconds into a tuple of (hours, minutes, seconds)"""

    hours = time_difference // 3600
    minutes = (time_difference // 60) % 60
    seconds = time_difference % 60

    return hours, minutes, seconds

def save_model(path, epoch, model, optimizer=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch,
            'state_dict': state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)

def split_data(features, encoding, label, ratio):

    num = features.shape[0]
    train_num = math.ceil(num * ratio)

    train_features = features[0: train_num, :, :]
    train_encoding = encoding[0: train_num, :, :]
    train_label = label[0: train_num, :]

    val_features = features[train_num: num, :, :]
    val_encoding = encoding[train_num: num, :, :]
    val_label = label[train_num: num, :]

    return train_features, train_encoding, train_label, val_features, val_encoding, val_label

def split_data(features, encoding, label, ratio):

    num = features.shape[0]
    train_num = math.ceil(num * ratio)

    train_features = features[0: train_num, :, :]
    train_encoding = encoding[0: train_num, :, :]
    train_label = label[0: train_num, :]

    val_features = features[train_num: num, :, :]
    val_encoding = encoding[train_num: num, :, :]
    val_label = label[train_num: num, :]

    return train_features, train_encoding, train_label, val_features, val_encoding, val_label

def split_data(features, encoding, label, ratio):

    num = features.shape[0]
    train_num = math.ceil(num * ratio)

    train_features = features[0: train_num, :, :]
    train_encoding = encoding[0: train_num, :, :]
    train_label = label[0: train_num, :]

    val_features = features[train_num: num, :, :]
    val_encoding = encoding[train_num: num, :, :]
    val_label = label[train_num: num, :]

    return train_features, train_encoding, train_label, val_features, val_encoding, val_label

# make sure each variable has one samples
def split_train(features, encoding, label, ratio):
    num_sub = features.shape[1]
    l_sub = features.shape[2]
    tar = torch.squeeze(label)
    all_class = torch.max(tar) + 1
    num_each_class = torch.bincount(tar)
    g = (num_each_class * ratio).int()
    h = num_each_class - g
    a = 0
    train_features, train_encoding, train_label = torch.empty(0, num_sub, l_sub), torch.empty(0, num_sub,
                                                                                              3), torch.empty(0,
                                                                                                              1)
    val_features, val_encoding, val_label = torch.empty(0, num_sub, l_sub), torch.empty(0, num_sub, 3), torch.empty(0,
                                                                                                                    1)
    for k in range(0, all_class):
        train_features = torch.cat((train_features, features[a: g[k] + a, :, :]), dim=0)
        train_encoding = torch.cat((train_encoding, encoding[a: g[k] + a, :, :]), dim=0)
        train_label = torch.cat((train_label, label[a: g[k] + a, :]), dim=0)

        val_features = torch.cat((val_features, features[g[k] + a: g[k] + a + h[k], :, :]), dim=0)
        val_encoding = torch.cat((val_encoding, encoding[g[k] + a: g[k] + a + h[k], :, :]), dim=0)
        val_label = torch.cat((val_label, label[g[k] + a: g[k] + a + h[k], :]), dim=0)
        a = g[k] + a + h[k]

    return train_features, train_encoding, train_label, val_features, val_encoding, val_label

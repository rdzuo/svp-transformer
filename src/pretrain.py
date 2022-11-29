import torch.utils.data as Data
from utils import split_train

def drop_tokens(embeddings, word_dropout):
    batch, length, size = embeddings.size()
    mask = embeddings.new_empty(batch, length)
    mask = mask.bernoulli_(1 - word_dropout)
    features = embeddings * mask.unsqueeze(-1).expand_as(embeddings).float()
    mask = mask.bool()
    mask = mask.unsqueeze(-1).expand(batch, length, size)
    return features, embeddings, mask

def split_pretrain(train_features, train_label, train_encoding, task, split_unlabel):
    # origin_task
    if (split_unlabel == False) and (task == 'classification'):
        train_dataset = Data.TensorDataset(train_features, train_label, train_encoding)
    # pretrain_task : pretrain
    elif (split_unlabel == False) and (task == 'pretrain'):
        train_features, target, mask = drop_tokens(train_features, word_dropout= 0.1)   # word_dropout is the part of masked
        train_dataset = Data.TensorDataset(train_features, target, train_encoding, mask)
    else:
        train_features, train_encoding, train_label, bu, xu, yao = split_train(train_features, train_encoding, train_label, 0.4)
        train_dataset = Data.TensorDataset(train_features, train_label, train_encoding)
    return train_dataset



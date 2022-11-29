import time
import torch.utils.data as Data
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import sys
import logging
from copy import deepcopy

logging.basicConfig(filename='new.log', filemode='w', format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

from local_features import *
from model import *
from running import *
from loss import get_loss_module
from optimizers import get_optimizer
from setting import config
from utils import split_data, split_train
from pretrain import split_pretrain

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', type=str, default="data/raw/",
                    help='the path of dataset')
parser.add_argument('--dataset', type=str, default="BasicMotions",
                    help='the name of dataset')
parser.add_argument('--use_preprocess', type=bool, default=True,
                    help='use preprocess data or not')
parser.add_argument('--load_model', type=bool, default=False,
                    help='load the trained model')
parser.add_argument('--test_only', type=bool, default=False,
                    help='only test')
args = parser.parse_args()

# set the random seed to be fixed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


total_start_time = time.time()

dataset = args.dataset

# load raw data
train_data, test_data, train_label, test_label, dim_length, dim_v, nclass = load_raw_ts(args.data_path, args.dataset)

# define window_size
default_value = 900
window_size = get_window(dim_length)

# number of center in each window
num_center_window = [int((3*default_value)/(6*dim_v)), int((2*default_value)/(6*dim_v)), int((1*default_value)/(6*dim_v))]
tol_sub = dim_v * sum(num_center_window)

# preprocess the data to find local features

if not args.use_preprocess:
    dataset_center = cluster_variable_window(train_data, window_size, dim_v, num_center_window)
    train_features, train_encoding = get_local_features(train_data, dataset_center, tol_sub)
    save_tensor_as_numpy(train_features, train_encoding, dataset, "train")
    test_features, test_encoding = get_local_features(test_data, dataset_center, tol_sub)
    save_tensor_as_numpy(test_features, test_encoding, dataset, "test")

# load the prepared-data
train = np.load("data/preprocess/" + args.dataset + "/" + "train.npz")
test = np.load("data/preprocess/" + args.dataset + "/" + "test.npz")

train_features = torch.from_numpy(train["arr_0"])  # number of instance, number of local shapes, length of instance
train_encoding = torch.from_numpy(train["arr_1"])  # number of instance, number of local shapes, 4 [length of subsequence,start,end,variable]

test_features = torch.from_numpy(test["arr_0"])
test_encoding = torch.from_numpy(test["arr_1"])

# preprocess encoding
train_encoding = preprocess(train_encoding, dim_v, dim_length)
test_encoding = preprocess(test_encoding, dim_v, dim_length)

# split the train dataset to train dataset and validation dataset
if config['split_ratio'] == 1:
    val_features = test_features
    val_encoding = test_encoding
    val_label = test_label
else:
    train_features, train_encoding, train_label, val_features, val_encoding, val_label = split_train(train_features, train_encoding, train_label, config['split_ratio'])

# check cuda
device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

# build model
model = model_factory(config, train_features, nclass)

# print model and parameters
logger.info("Model:\n{}".format(model))
logger.info("Total number of parameters: {}".format(utils.count_parameters(model)))
logger.info("Trainable parameters: {}".format(utils.count_parameters(model, trainable=True)))


# set start epoch and lr
start_epoch = 0
lr_step = 0  # current step index of `lr_step`
lr = config['lr']  # current learning step

if config['global_reg']:
    weight_decay = config['l2_reg']
    output_reg = None
else:
    weight_decay = 0
    output_reg = config['l2_reg']


# set optimizer and loss function
optim_class = get_optimizer(config['optimizer'])
optimizer = optim_class(model.parameters(), lr=config['lr'], weight_decay=weight_decay)
loss_module = get_loss_module(config)
collate_fn, runner_class = pipeline_factory(config['task'])

model.to(device)

# load trained model
if args.load_model == True:
    model_path = config['load_dir']
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    state_dict = deepcopy(checkpoint['state_dict'])
    start_epoch = checkpoint['epoch']
    print('Loaded model from {}. Epoch: {}'.format(model_path, checkpoint['epoch']))
    if config['change_output']:
        start_epoch = 0
        for key, val in checkpoint['state_dict'].items():
            if key.startswith('output_layer'):
                state_dict.pop(key)
    model.load_state_dict(state_dict, strict=False)
    optimizer.load_state_dict



# only test without training
if args.test_only == True:
    test_dataset = Data.TensorDataset(test_features, test_label, test_encoding)
    test_loader = Data.DataLoader(dataset=test_dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=0,
                                  pin_memory=True,
                                  collate_fn=lambda x: collate_fn(x))
    test_evaluator = runner_class(model, test_loader, device, loss_module,
                                  print_interval=config['print_interval'], console=config['console'])
    with torch.no_grad():
        aggr_metrics_test, per_batch_test = test_evaluator.evaluate(start_epoch, keep_all=True)

    print_str = 'Test Summary: '
    for k, v in aggr_metrics_test.items():
        print_str += '{}: {:8f} | '.format(k, v)
    logger.info(print_str)
    sys.exit()

train_dataset = split_pretrain(train_features, train_label, train_encoding, config['task'], config['split_unlabel'])

# train_dataset = Data.TensorDataset(train_features, train_label, train_encoding)

train_loader = Data.DataLoader(dataset=train_dataset,
                               batch_size=config['batch_size'],
                               shuffle=True,
                               num_workers=0,
                               pin_memory=True,
                               collate_fn=lambda x: collate_fn(x))

val_dataset = Data.TensorDataset(val_features, val_label, val_encoding)

val_loader = Data.DataLoader(dataset=val_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=0,
                             pin_memory=True,
                             collate_fn=lambda x: collate_fn(x))

trainer = runner_class(model, train_loader, device, loss_module, optimizer, l2_reg=output_reg,
                       print_interval=config['print_interval'], console=config['console'])

val_evaluator = runner_class(model, val_loader, device, loss_module,
                             print_interval=config['print_interval'], console=config['console'])

tensorboard_writer = SummaryWriter('experiments/_fromScratch_2021-11-24_15-56-09_Au1/tb_summaries')

best_value = -1e16  # initialize with +inf or -inf depending on key metric
metrics = []  # (for validation) list of lists: for each epoch, stores metrics like loss, ...
best_metrics = {}

total_epoch_time = 0
total_eval_time = 0

for epoch in tqdm(range(start_epoch + 1, config["epochs"] + 1), desc='Training Epoch', leave=False):
    # mark = epoch if config['save_all'] else 'last'
    mark = 'last'
    epoch_start_time = time.time()
    aggr_metrics_train = trainer.train_epoch(epoch)  # dictionary of aggregate epoch metrics
    epoch_runtime = time.time() - epoch_start_time
    print()
    print_str = 'Epoch {} Training Summary: '.format(epoch)
    for k, v in aggr_metrics_train.items():
        tensorboard_writer.add_scalar('{}/train'.format(k), v, epoch)
        print_str += '{}: {:8f} | '.format(k, v)
    logger.info(print_str)
    logger.info("Epoch runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(epoch_runtime)))
    total_epoch_time += epoch_runtime
    avg_epoch_time = total_epoch_time / (epoch - start_epoch)
    avg_batch_time = avg_epoch_time / config['batch_size']
    avg_sample_time = avg_epoch_time / train_data.shape[0]
    logger.info("Avg epoch train. time: {} hours, {} minutes, {} seconds".format(*utils.readable_time(avg_epoch_time)))
    logger.info("Avg batch train. time: {} seconds".format(avg_batch_time))
    logger.info("Avg sample train. time: {} seconds".format(avg_sample_time))

    # evaluate if first or last epoch or at specified interval
    if config['task'] == 'classification':

        if (epoch == config["epochs"]) or (epoch == start_epoch + 1) or (epoch % config['val_interval'] == 0):
            aggr_metrics_val, best_metrics, best_value = validate(val_evaluator, tensorboard_writer, config,
                                                              best_metrics, best_value, epoch)
            metrics_names, metrics_values = zip(*aggr_metrics_val.items())
            metrics.append(list(metrics_values))


    utils.save_model(os.path.join(config['save_dir'], 'model_{}.pth'.format(mark)), epoch, model, optimizer)

    logger.info('Best {} was {}. Other metrics: {}'.format(config['key_metric'], best_value, best_metrics))
    logger.info('All Done!')

    total_runtime = time.time() - total_start_time
    logger.info("Total runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(total_runtime)))



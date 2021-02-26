import os.path
import sqlite3
import pandas as pd
import numpy as np
import io

from typing import List

from models import *

# the values of these command line arguments are used to define the name of the experiments
# you can add more names in this list
EXPERIMENT_PARAMETERS = ["model", "type", "N", "flip", "restrict", "sgsize", "fixparams", "augment", "F", "sigma", "interpolation"]


########################################################################################################################
# Utilites to store and retrieve results of the experiments
########################################################################################################################


def update_logs(logs: pd.DataFrame, path: str):
    conn = sqlite3.connect(path)
    logs.to_sql("logs", conn, if_exists="append")
    conn.close()


def retrieve_logs(path: str) -> pd.DataFrame:
    conn = sqlite3.connect(path)
    logs = pd.read_sql_query("select * from logs;", conn)
    conn.close()
    
    return logs


# create data type in sqlite to store numpy arrays

# convert array to binary to store it in sqlite

def encode_array2binary(x: np.ndarray):
    binary_buffer = io.BytesIO()
    np.save(binary_buffer, x)
    binary_buffer.seek(0)
    y = binary_buffer.read()
    y = sqlite3.Binary(y)
    return y

sqlite3.register_adapter(np.ndarray, encode_array2binary)

# recover array from binary encoding in the sqlite database

def decode_binary2array(y):
    binary_buffer = io.BytesIO(y)
    binary_buffer.seek(0)
    x = np.load(binary_buffer)
    return x

sqlite3.register_converter("array", decode_binary2array)

##########################################################


def update_confusion(confusion_matrix: np.array, path: str):
    assert confusion_matrix.shape[0] == confusion_matrix.shape[1]
    
    create = '''CREATE TABLE IF NOT EXISTS confusions (matrix array)'''
    
    insert = '''INSERT INTO confusions (matrix) VALUES (?)'''
    
    conn = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES)
    cursor = conn.cursor()
    cursor.execute(create)
    cursor.execute(insert, (confusion_matrix,))
    
    conn.commit()
    conn.close()


def retrieve_confusion(path: str) -> List[np.array]:
    conn = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES)
    cursor = conn.cursor()
    
    cursor.execute("select * from confusions")
    arrays = cursor.fetchall()
    conn.close()
    
    arrays = [a[0] for a in arrays]
    
    return arrays


########################################################################################################################
# Utilites to build paths and names in a standard way
########################################################################################################################


def exp_name(config):
    config = vars(config)
    return '_'.join([str(config[p]) for p in EXPERIMENT_PARAMETERS])


def out_path(config):
    path = 'results/{}'.format(config.dataset)
    if config.reshuffle:
        path += "(shuffled)"
    if config.augment:
        path += "_(train_augmentation)"
    if not config.earlystop:
        path += "_(full_train)"
    return path


def plot_path(config):
    return os.path.join(out_path(config), exp_name(config) + ".svg")


def backup_path(config):
    backup_folder = os.path.join(out_path(config), exp_name(config))
    return os.path.join(backup_folder, f"_{config.seed}.model")


def logs_path(config):
    return os.path.join(out_path(config), exp_name(config) + ".db")


def build_model(config, n_inputs, n_outputs):
    # SFCNN VARIANTS
    if config.model == 'E2SFCNN':
        model = E2SFCNN(n_inputs, n_outputs, restrict=config.restrict, N=config.N, fco=config.F, J=config.J,
                        sigma=config.sigma, fix_param=config.fixparams, sgsize=config.sgsize, flip=config.flip)
    elif config.model == 'E2SFCNN_QUOT':
        model = E2SFCNN_QUOT(n_inputs, n_outputs, restrict=config.restrict, N=config.N, fco=config.F, J=config.J,
                             sigma=config.sigma, sgsize=config.sgsize, flip=config.flip)
    elif config.model == 'EXP':
        model = ExpE2SFCNN(n_inputs, n_outputs, layer_type=config.type, restrict=config.restrict, N=config.N,
                           fix_param=config.fixparams, fco=config.F, J=config.J, sigma=config.sigma,
                           deltaorth=config.deltaorth, antialias=config.antialias, sgsize=config.sgsize,
                           flip=config.flip)
    elif config.model == 'CNN':
        model = ExpCNN(n_inputs, n_outputs, fix_param=config.fixparams, deltaorth=config.deltaorth)
    elif config.model == "wrn16_8_stl":
        model = wrn16_8_stl(num_classes=n_outputs, deltaorth=config.deltaorth)
    elif config.model == "e2wrn16_8_stl":
        model = e2wrn16_8_stl(N=config.N, r=config.restrict, num_classes=n_outputs, sigma=config.sigma, F=config.F,
                               deltaorth=config.deltaorth, fixparams=config.fixparams)
    elif config.model == "e2wrn28_10":
        model = e2wrn28_10(N=config.N, r=config.restrict, num_classes=n_outputs, sigma=config.sigma, F=config.F,
                            deltaorth=config.deltaorth, fixparams=config.fixparams)
    elif config.model == "e2wrn28_7":
        model = e2wrn28_7(N=config.N, r=config.restrict, num_classes=n_outputs, sigma=config.sigma, F=config.F,
                           deltaorth=config.deltaorth, fixparams=config.fixparams)
    elif config.model == "e2wrn28_10R":
        model = e2wrn28_10R(N=config.N, r=config.restrict, num_classes=n_outputs, sigma=config.sigma, F=config.F,
                             deltaorth=config.deltaorth, fixparams=config.fixparams)
    elif config.model == "e2wrn28_7R":
        model = e2wrn28_7R(N=config.N, r=config.restrict, num_classes=n_outputs, sigma=config.sigma, F=config.F,
                            deltaorth=config.deltaorth, fixparams=config.fixparams)
    else:
        raise ValueError("Model selected ({}) not recognized!".format(config.model))
    
    return model


########################################################################################################################
# utilites to build dataloaders
########################################################################################################################

from datasets.mnist_rot import data_loader_mnist_rot
from datasets.mnist_fliprot import data_loader_mnist_fliprot
from datasets.mnist12k import data_loader_mnist12k
from datasets.cifar10 import data_loader_cifar10
from datasets.cifar100 import data_loader_cifar100
from datasets.STL10 import data_loader_stl10
from datasets.STL10 import data_loader_stl10frac


def build_dataloaders(dataset, batch_size, num_workers, augment, validation=True, reshuffle=False,
                      eval_batch_size=None, interpolation=2):
    if eval_batch_size is None:
        eval_batch_size = batch_size
        
    if dataset == "mnist_rot":
        
        if validation:
            if reshuffle:
                seed = np.random.randint(0, 100000)
            else:
                seed = None
            train_loader, _, _ = data_loader_mnist_rot.build_mnist_rot_loader("train",
                                                                              batch_size,
                                                                              rot_interpol_augmentation=augment,
                                                                              interpolation=interpolation,
                                                                              reshuffle_seed=seed)
            valid_loader, _, _ = data_loader_mnist_rot.build_mnist_rot_loader("valid",
                                                                              eval_batch_size,
                                                                              rot_interpol_augmentation=False,
                                                                              interpolation=interpolation,
                                                                              reshuffle_seed=seed)
        else:
            train_loader, _, _ = data_loader_mnist_rot.build_mnist_rot_loader("trainval",
                                                                              batch_size,
                                                                              rot_interpol_augmentation=augment,
                                                                              interpolation=interpolation,
                                                                              reshuffle_seed=None)
            valid_loader = False
        
        test_loader, n_inputs, n_outputs = data_loader_mnist_rot.build_mnist_rot_loader("test",
                                                                                        eval_batch_size,
                                                                                        rot_interpol_augmentation=False)
    
    elif dataset == "mnist_fliprot":
        
        if validation:
            if reshuffle:
                seed = np.random.randint(0, 100000)
            else:
                seed = None
            
            train_loader, _, _ = data_loader_mnist_fliprot.build_mnist_rot_loader("train",
                                                                                  batch_size,
                                                                                  rot_interpol_augmentation=augment,
                                                                                  interpolation=interpolation,
                                                                                  reshuffle_seed=seed)
            valid_loader, _, _ = data_loader_mnist_fliprot.build_mnist_rot_loader("valid",
                                                                                  eval_batch_size,
                                                                                  rot_interpol_augmentation=False,
                                                                                  interpolation=interpolation,
                                                                                  reshuffle_seed=seed)
        else:
            train_loader, _, _ = data_loader_mnist_fliprot.build_mnist_rot_loader("trainval",
                                                                                  batch_size,
                                                                                  rot_interpol_augmentation=augment,
                                                                                  interpolation=interpolation,
                                                                                  reshuffle_seed=None)
            valid_loader = False
        
        test_loader, n_inputs, n_outputs = data_loader_mnist_fliprot.build_mnist_rot_loader("test",
                                                                                            eval_batch_size,
                                                                                            rot_interpol_augmentation=False)
    elif dataset == "mnist12k":
        
        if validation:
            if reshuffle:
                seed = np.random.randint(0, 100000)
            else:
                seed = None
            train_loader, _, _ = data_loader_mnist12k.build_mnist12k_loader("train",
                                                                            batch_size,
                                                                            rot_interpol_augmentation=augment,
                                                                            interpolation=interpolation,
                                                                            reshuffle_seed=seed)
            valid_loader, _, _ = data_loader_mnist12k.build_mnist12k_loader("valid",
                                                                            eval_batch_size,
                                                                            rot_interpol_augmentation=False,
                                                                            interpolation=interpolation,
                                                                            reshuffle_seed=seed)
        else:
            train_loader, _, _ = data_loader_mnist12k.build_mnist12k_loader("trainval",
                                                                            batch_size,
                                                                            rot_interpol_augmentation=augment,
                                                                            interpolation=interpolation,
                                                                            reshuffle_seed=None)
            valid_loader = False
        
        test_loader, n_inputs, n_outputs = data_loader_mnist12k.build_mnist12k_loader("test",
                                                                                      eval_batch_size,
                                                                                      # rot_interpol_augmentation=False
                                                                                      # interpolation=interpolation,
                                                                                      )
    elif dataset == "STL10":
        train_loader, valid_loader, test_loader, n_inputs, n_outputs = data_loader_stl10.build_stl10_loaders(
            batch_size,
            eval_batch_size,
            validation=validation,
            augment=augment,
            num_workers=num_workers,
            reshuffle=reshuffle
        )
    elif dataset == "STL10cif":
        train_loader, valid_loader, test_loader, n_inputs, n_outputs = data_loader_stl10.build_stl10cif_loaders(
            batch_size,
            eval_batch_size,
            validation=validation,
            augment=augment,
            num_workers=num_workers,
            reshuffle=reshuffle
        )
    elif dataset.startswith("STL10|"):
        size = int(dataset.split("|")[1])
        train_loader, valid_loader, test_loader, n_inputs, n_outputs = data_loader_stl10frac.build_stl10_frac_loaders(
            size,
            batch_size,
            eval_batch_size,
            validation=validation,
            augment=augment,
            num_workers=num_workers,
            reshuffle=reshuffle
        )
    elif dataset.startswith("STL10cif|"):
        size = int(dataset.split("|")[1])
        train_loader, valid_loader, test_loader, n_inputs, n_outputs = data_loader_stl10frac.build_stl10cif_frac_loaders(
            size,
            batch_size,
            eval_batch_size,
            validation=validation,
            augment=augment,
            num_workers=num_workers,
            reshuffle=reshuffle
        )
    elif dataset == "cifar10":
        train_loader, valid_loader, test_loader, n_inputs, n_outputs = data_loader_cifar10.build_cifar10_loaders(
            batch_size,
            eval_batch_size,
            validation=validation,
            augment=augment,
            num_workers=num_workers,
            reshuffle=reshuffle
        )
    elif dataset == "cifar100":
        train_loader, valid_loader, test_loader, n_inputs, n_outputs = data_loader_cifar100.build_cifar100_loaders(
            batch_size,
            eval_batch_size,
            validation=validation,
            augment=augment,
            num_workers=num_workers,
            reshuffle=reshuffle
        )
    else:
        raise ValueError("Dataset '{}' not recognized!".format(dataset))
    
    dataloaders = {"train": train_loader, "valid": valid_loader, "test": test_loader}
    return dataloaders, n_inputs, n_outputs


########################################################################################################################
# utilites to build experiments' args parser
########################################################################################################################

import argparse

SHOW_PLOT = False
SAVE_PLOT = True

RESHUFFLE = False
AUGMENT_TRAIN = False

LEARNING_RATE = 1e-4
BATCH_SIZE = 64
EPOCHS = 40

PLOT_FREQ = 100

EVAL_FREQ = 100

BACKUP = False
BACKUP_FREQ = -1

ADAPT_LR = False


def none_or_float(value):
    if value == 'None':
        return None
    return float(value)


def args_exp_parameters(parser):
    ######## EXPERIMENT'S PARAMETERS ########

    # Dataset params
    parser.add_argument('--dataset', type=str, help='The name of the dataset to use')
    parser.add_argument('--augment', dest="augment", action="store_true",
                        help='Augment the training set with rotated images')
    parser.set_defaults(augment=AUGMENT_TRAIN)
    parser.add_argument('--interpolation', type=int, default=2,
                        help='Type of interpolation to use for data augmentation')
    
    parser.add_argument('--reshuffle', dest="reshuffle", action="store_true",
                        help='Reshuffle train and valid splits instead of using the default split')
    parser.set_defaults(reshuffle=RESHUFFLE)
    parser.add_argument('--workers', type=int, default=8, help='Number of jobs to load the dataset')
    parser.add_argument('--time_limit', type=int, default=None, help='Maximum time limit for training (in Minutes)')
    parser.add_argument('--verbose', type=int, default=2, help='Verbose Level')
    
    # Model params
    parser.add_argument('--model', type=str, help='The name of the model to use')
    parser.add_argument('--type', type=str, default=None, help='Type of fiber for the EXP model')
    parser.add_argument('--N', type=int, help='Size of cyclic group for GCNN and maximum frequency for HNET')
    parser.add_argument('--F', type=none_or_float, default=None,
                        help='Frequency cut-off: maximum frequency at radius "r" is "F*r"')
    parser.add_argument('--sigma', type=none_or_float, default=None,
                        help='Width of the rings building the bases (std of the gaussian window)')
    parser.add_argument('--J', type=int, default=None,
                        help='Number of additional frequencies in the interwiners of finite groups')
    parser.add_argument('--restrict', type=int, default=-1, help='Layer where to restrict SFCNN from E(2) to SE(2)')
    parser.add_argument('--sgsize', type=int, default=None,
                        help='Number of rotations in the subgroup to restrict to in the EXP e2sfcnn models')
    parser.add_argument('--flip', dest="flip", action="store_true",
                        help='Use also reflection equivariance in the EXP model')
    parser.set_defaults(flip=False)
    parser.add_argument('--fixparams', dest="fixparams", action="store_true",
                        help='Keep the number of parameters of the model fixed by adjusting its topology')
    parser.set_defaults(fixparams=False)
    parser.add_argument('--deltaorth', dest="deltaorth", action="store_true",
                        help='Use delta orthogonal initialization in conv layers')
    parser.set_defaults(deltaorth=False)
    parser.add_argument('--antialias', type=float, default=0.,
                        help='Std for the gaussian blur in the max-pool layer. If zero (default), standard maxpooling is performed')
    
    # Training params
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Number of examples to process in a batch training')
    parser.add_argument('--eval_batch_size', type=int, default=None,
                        help='Number of examples to process in a batch during test. By default (None), it is the same as batch_size.')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of training epochs')
    parser.add_argument('--steps_per_epoch', type=int, default=-1,
                        help='Number of batches to process during each epoch. By default, it processes all batches')
    parser.add_argument('--no_earlystop', dest="earlystop", action="store_false",
                        help="Don't split the training set to build a validation set for early stopping but train on the union of validation and training set")
    parser.set_defaults(earlystop=True)
    parser.add_argument('--valid_metric', type=str, default="accuracy",
                        help='Metric on the validation set to use for early stopping')
    parser.add_argument('--eval_test', dest="eval_test", action="store_true",
                        help="Evaluate and logs model's performance also on the test set when doing validation during training (not used for early stopping)")
    parser.set_defaults(eval_test=False)
    parser.add_argument('--accumulate', type=int, default=1,
                        help='During training, accumulate the gradinet of a number of batches before optimizing the model. Useful for large models when a single batch does not fit in memory. (By default, only 1 batch is accumulated)')
    
    parser.add_argument('--optimizer', type=str, default="sfcnn", choices=["sfcnn", "SGD", "Adam"],
                        help='Optimize to use')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum (used only if optimizer = SGD)')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='Weight Decay (used only if optimizer = SGD or Adam)')
    
    parser.add_argument('--adapt_lr', type=str, default=None,
                        help='Adaptive learning rate scheduler to sue (default = none). '
                             'Available choices: "exponential" or "validation" (the last one requires earlystop)')
    parser.add_argument('--lr_decay_start', type=int, default=15, help='Starting epoch for the adaptive lr scheduler')
    parser.add_argument('--lr_decay_factor', type=float, default=0.8, help='Decay factor for the adaptive lr scheduler')
    parser.add_argument('--lr_decay_epoch', type=int, default=1,
                        help='Period (in number of epochs) for the adaptive lr scheduler')
    parser.add_argument('--lr_decay_schedule', type=int, nargs='+', default=None,
                        help='Epochs when lr should be decayed for the adaptive lr scheduler '
                             '(alternative for lr_decay_epoch)')
    
    # Regularization params
    parser.add_argument('--l1', dest="l1", action="store_true",
                        help="Use L1L2 regularization as in SFCNN paper")
    parser.set_defaults(l1=False)
    
    parser.add_argument('--lamb_conv_L1',
                        type=float,
                        default=1e-7,
                        help='gain of L1 loss for steerable layer variables')
    parser.add_argument('--lamb_conv_L2',
                        type=float,
                        default=1e-7,
                        help='gain of L2 loss for steerable layer variables')
    parser.add_argument('--lamb_fully_L1',
                        type=float,
                        default=1e-8,
                        help='gain of L1 loss for fully connected layer variables')
    parser.add_argument('--lamb_fully_L2',
                        type=float,
                        default=1e-8,
                        help='gain of L2 loss for fully connected layer variables')
    parser.add_argument('--lamb_softmax_L2',
                        type=float,
                        default=0,
                        help='gain of L2 loss for fully connected layer variables')
    parser.add_argument('--lamb_bn_L1',
                        type=float,
                        default=0,
                        help='gain of L1 loss for batchnorm weights')
    parser.add_argument('--lamb_bn_L2',
                        type=float,
                        default=0,
                        help='gain of L2 loss for batchnorm weights')
    
    # Other experiment's parameters
    parser.add_argument('--eval_frequency', type=int, default=EVAL_FREQ,
                        help='Evaluation frequency (counts iterations if positive, epochs if negative. Use -1 to evaluate at the end of each epoch)')
    
    parser.add_argument('--backup_model', dest="backup_model", action="store_true", help='Backup the model in a file')
    parser.set_defaults(backup_model=BACKUP)
    
    parser.add_argument('--plot_frequency', type=int, default=PLOT_FREQ,
                        help='Plot frequency (counts iterations if positive, epochs if negative. Use -1 to plot at the end of each epoch)')
    parser.add_argument('--store_plot', dest="store_plot", action="store_true",
                        help="Store the plot")
    parser.set_defaults(store_plot=False)
    parser.add_argument('--show', dest="show", action="store_true",
                        help="Show the plot during training")
    parser.set_defaults(show=False)
    
    parser.add_argument('--backup_frequency', type=int, default=BACKUP_FREQ,
                        help='Backup frequency (counts iterations if positive, epochs if negative. Use -1 to backup at the end of each epoch)')
    
    return parser


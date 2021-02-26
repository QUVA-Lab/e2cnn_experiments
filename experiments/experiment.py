import numpy as np
import torch
import torch.nn as nn

import e2cnn.nn as enn

import pandas as pd
import argparse
import os
import datetime

import plot_exps
import utils
import optimizers_L1L2

from sklearn.metrics import confusion_matrix

import matplotlib

if "DISPLAY" not in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

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


def compute_confusion_matrix(predictions, targets, labels):
    if predictions.shape[1] > 1:
        predictions = predictions.argmax(dim=1)
    else:
        predictions = (predictions > 0.)
    
    return confusion_matrix(targets.cpu().numpy(), predictions.cpu().numpy(), labels)


def accuracy(predictions, targets):
    if predictions.shape[1] > 1:
        predictions = predictions.argmax(dim=1)
    else:
        predictions = (predictions > 0.)
    
    predictions = predictions.to(dtype=targets.dtype)
    accuracy = float((targets == predictions).sum()) / predictions.numel()
    return accuracy


def build_optimizer_sfcnn(model, config):
    # optimizer as in "Learning Steerable Filters for Rotation Equivariant CNNs"
    # https://arxiv.org/abs/1711.07289
    
    # split up parameters into groups, named_parameters() returns tuples ('name', parameter)
    # each group gets its own regularization gain
    batchnormLayers = [m for m in model.modules() if isinstance(m,
                                                                     (nn.modules.batchnorm.BatchNorm1d,
                                                                      nn.modules.batchnorm.BatchNorm2d,
                                                                      nn.modules.batchnorm.BatchNorm3d,
                                                                      enn.NormBatchNorm,
                                                                      enn.GNormBatchNorm,
                                                                      )
                                                                )]
    linearLayers = [m for m in model.modules() if isinstance(m, nn.modules.linear.Linear)]
    convlayers = [m for m in model.modules() if isinstance(m, (nn.Conv2d, enn.R2Conv))]
    weights_conv = [p for m in convlayers for n, p in m.named_parameters() if n.endswith('weights') or n.endswith("weight")]
    biases = [p for n, p in model.named_parameters() if n.endswith('bias')]
    weights_bn = [p for m in batchnormLayers for n, p in m.named_parameters()
                  if n.endswith('weight') or n.split('.')[-1].startswith('weight')
                  ]
    weights_fully = [p for m in linearLayers for n, p in m.named_parameters() if n.endswith('weight')]
    # CROP OFF LAST WEIGHT !!!!! (classification layer)
    weights_fully, weights_softmax = weights_fully[:-1], [weights_fully[-1]]
    print("SFCNN optimizer")
    for n, p in model.named_parameters():
        if p.requires_grad and not n.endswith(('weight', 'weights', 'bias')):
            raise Exception('named parameter encountered which is neither a weight nor a bias but `{:s}`'.format(n))
    param_groups = [dict(params=weights_conv, lamb_L1=config.lamb_conv_L1, lamb_L2=config.lamb_conv_L2, weight_decay=config.lamb_conv_L2),
                    dict(params=weights_bn, lamb_L1=config.lamb_bn_L1, lamb_L2=config.lamb_bn_L2, weight_decay=config.lamb_bn_L2),
                    dict(params=weights_fully, lamb_L1=config.lamb_fully_L1, lamb_L2=config.lamb_fully_L2, weight_decay=config.lamb_fully_L2),
                    dict(params=weights_softmax, lamb_L1=0, lamb_L2=config.lamb_softmax_L2, weight_decay=config.lamb_softmax_L2),
                    dict(params=biases, lamb_L1=0, lamb_L2=0, weight_decay=0)]
    if config.l1:
        return optimizers_L1L2.Adam(param_groups, lr=config.lr, betas=(0.9, 0.999))
    else:
        return torch.optim.Adam(param_groups, lr=config.lr, betas=(0.9, 0.999))


class Experiment:
    
    def __init__(self, config):
        super(Experiment, self).__init__()
        
        self.seed = config.seed
        
        self._verbose = config.verbose
        
        self.earlystop = config.earlystop
        self.eval_test = config.eval_test
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        
        print("DEVICE:", self.device)
        
        self.logs = pd.DataFrame(data=[], columns=["seed", "split", "iteration", "accuracy", "loss"])
        
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        # build the datasets and the train, validation and test loaders
        self._dataloaders, n_inputs, n_outputs = utils.build_dataloaders(
            config.dataset,
            config.batch_size,
            config.workers,
            config.augment,
            config.earlystop,
            config.reshuffle,
            eval_batch_size=config.eval_batch_size,
            interpolation=config.interpolation,
        )
        
        print("datasets built")
        
        if n_outputs == 2:
            n_outputs = 1
            self._loss_function = torch.nn.BCEWithLogitsLoss()
        else:
            self._loss_function = torch.nn.CrossEntropyLoss()
        
        self.n_outputs = n_outputs
        
        # build the model
        self.model = utils.build_model(config, n_inputs, n_outputs)
        
        self.outpath = utils.out_path(config)
        os.makedirs(self.outpath, exist_ok=True)
        
        self.expname = utils.exp_name(config)
        
        # visualization parameters
        self._show = config.show
        self._plot_frequency = config.plot_frequency
        
        if config.store_plot or config.show:
            self._visualization = plt.subplots(1, 2, figsize=(10, 4))
        else:
            self._visualization = None
        
        if config.store_plot:
            self.plotpath = utils.plot_path(config)
        else:
            self.plotpath = None
        
        # backup model parameters
        self._backup_frequency = config.backup_frequency
        self._backup_model = config.backup_model
        self.modelpath = utils.backup_path(config)
        if self._backup_model:
            os.makedirs(os.path.dirname(self.modelpath), exist_ok=True)

        # training configuration
        self.epochs = config.epochs
        self._eval_frequency = config.eval_frequency
        self.batch_size = config.batch_size
        self.accumulate = config.accumulate
        self.steps_per_epoch = config.steps_per_epoch
        self._lr = config.lr
        
        self._lr_decay_start = config.lr_decay_start
        self._lr_decay_factor = config.lr_decay_factor
        self._lr_decay_epoch = config.lr_decay_epoch
        self._lr_decay_schedule = config.lr_decay_schedule
        if self._lr_decay_schedule is not None:
            self._lr_decay_epoch = None
            self._lr_decay_start = None
        
        self._lr_exp_steps = 0
        if config.optimizer == "sfcnn":
            # optimize as in "Learning Steerable Filters for Rotation Equivariant CNNs"
            # https://arxiv.org/abs/1711.07289
            self._optimizer = build_optimizer_sfcnn(self.model, config)
        elif config.optimizer == "Adam":
            self._optimizer = torch.optim.Adam(self.model.parameters(),
                                               lr=config.lr,
                                               weight_decay=config.weight_decay
                                               )
        elif config.optimizer == "SGD":
            self._optimizer = torch.optim.SGD(self.model.parameters(),
                                              lr=config.lr,
                                              momentum=config.momentum,
                                              weight_decay=config.weight_decay)
        self._adapt_lr_type = config.adapt_lr
        if config.adapt_lr == "exponential":
            self._adapt_lr = self._lr_scheduler_exponential_decay
        elif config.adapt_lr == "validation":
            assert config.earlystop
            self._lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self._optimizer,
                                                                            factor=self._lr_decay_factor,
                                                                            patience=self._lr_decay_epoch,
                                                                            verbose=self._verbose > 2,
                                                                            eps=1e-8,
                                                                            )
            self._adapt_lr = self._lr_scheduler.step
            
        elif config.adapt_lr is not None:
            raise ValueError()
        else:
            config.adapt_lr = None
        
        self._iteration = 0
        self._epoch = 0
        
        self.model.to(self.device)
        self.model = nn.DataParallel(self.model)
        
        if self.earlystop:
            assert config.valid_metric in ["loss", "accuracy"]
            self._valid_metric = config.valid_metric
        self._last_valid_metric = 1e+20
        self.best_valid_iteration = 0
        self.best_valid_loss = 1e+20
        self.best_valid_accuracy = 0
        self.best_state_dict = self.model.state_dict()
        
        self._time_limit = config.time_limit
        self._start_time = datetime.datetime.now()
        
        if self._verbose > 1:
            tot_param = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
            print("Total number of parameters:", tot_param)
        
        if self._verbose > 1:
            print("Starting: ", self._start_time)

    def log(self, accuracy, loss, split):
        row = [self.seed, split, self._iteration, accuracy, loss]
        self.logs.loc[len(self.logs)] = row
    
    def backup(self):
        if self._backup_model:
            torch.save(self.best_state_dict, self.modelpath)
    
    def train(self):
        train_len = len(self._dataloaders["train"])
        data_len = len(self._dataloaders["train"].dataset)
        
        actual_batch_size = self.batch_size * self.accumulate
        last_batch_size = data_len % actual_batch_size
        n_batches = data_len // actual_batch_size + (last_batch_size > 0)
        
        self._optimizer.zero_grad()
        
        epoch_iterations = 0
        cumulative_loss = 0
        cumulative_acc = 0
        for batch_idx, (x, t) in enumerate(self._dataloaders["train"]):
            if self._verbose > 3:
                print(f"\ttrain:{batch_idx}/{train_len}\t\t{datetime.datetime.now()}")
            
            batchsize = actual_batch_size if epoch_iterations < n_batches - 1 else last_batch_size
            
            self.model.train()
            
            x = x.to(self.device)
            t = t.to(self.device)
            
            y = self.model(x)
            
            loss = self._loss_function(y, t) * x.shape[0] / batchsize
            
            acc = accuracy(y.detach(), t.detach())
            
            cumulative_loss += loss.item()
            cumulative_acc += acc * x.shape[0] / batchsize
            
            loss.backward()
            
            del loss
            del y
            del x
            del t

            if (batch_idx + 1) % self.accumulate == 0 or batch_idx == train_len - 1:
                if self._verbose > 2:
                    print("Epoch {} | {}/{}; loss: {}; acc: {}".format(self._epoch,
                                                                       epoch_iterations,
                                                                       n_batches,
                                                                       cumulative_loss,
                                                                       cumulative_acc))

                self.log(cumulative_acc, cumulative_loss, "train")
                
                self._optimizer.step()
                self._optimizer.zero_grad()
            
                if self._eval_frequency > 0 and self._iteration % self._eval_frequency == 0:
                    self.valid()
                
                self._iteration += 1
                epoch_iterations += 1
                cumulative_loss = 0
                cumulative_acc = 0
                
                if self._backup_frequency > 0 and self._iteration % self._backup_frequency == 0:
                    self.backup()
                
                if self._plot_frequency > 0 and self._iteration % self._plot_frequency == 0:
                    self.plot()

                if self.steps_per_epoch > 0 and epoch_iterations >= self.steps_per_epoch:
                    break
    
    def test(self):
        if self._verbose > 0:
            print("\n")
            print("############################################ START TESTING ########################################")
        
        if self.earlystop:
            self.model.eval()
            self.model.load_state_dict(self.best_state_dict)
        
        acc, loss, conf_matrix = self.evaluate("test", confusion=True)
        
        self.conf_matrix = conf_matrix
        
        if self._verbose > 0:
            np.set_printoptions(precision=4, suppress=True, threshold=1000000, linewidth=1000000)
            print("##### ExperimentClassification [{}]".format(self.expname))
            print("##### TEST LOSS = {}".format(loss))
            print("##### TEST ACCURACY = {}".format(acc))
            print("###################################################################################################")
            print("# Confusion Matrix")
            print(conf_matrix)
            print("# Normalized Confusion Matrix")
            conf_matrix /= conf_matrix.sum(axis=1, keepdims=True)
            print(conf_matrix)
            print("###################################################################################################")
            print("\n")
    
    def valid(self):
        if self.earlystop:
            acc, loss = self.evaluate("valid")
            
            if self._verbose > 1:
                print('################################################################################')
                print('Evaluating [{}] on VALID| Epoch: {}; Iteration: {}; Accuracy: {}; Loss: {}'.format(self.expname,
                                                                                                          self._epoch,
                                                                                                          self._iteration,
                                                                                                          acc,
                                                                                                          loss))
                print('################################################################################')

            if self._valid_metric == "accuracy":
                _last_valid_metric = acc
                if acc > self.best_valid_accuracy:
                    self.best_valid_iteration = self._epoch
                    self.best_state_dict = self.model.state_dict()
                    
            elif self._valid_metric == "loss":
                _last_valid_metric = loss
                if loss < self.best_valid_loss:
                    self.best_valid_iteration = self._epoch
                    self.best_state_dict = self.model.state_dict()
            else:
                raise ValueError(self._valid_metric)
                
            if self._adapt_lr_type == "validation" and self._adapt_lr is not None:
                self._adapt_lr(_last_valid_metric)
            
            self.best_valid_loss = min(loss, self.best_valid_loss)
            self.best_valid_accuracy = max(acc, self.best_valid_accuracy)
        
        if self.eval_test:
            acc, loss = self.evaluate("test")
            if self._verbose > 1:
                print('################################################################################')
                print('Evaluating [{}] on TEST | Epoch: {}; Iteration: {}; Accuracy: {}; Loss: {}'.format(self.expname,
                                                                                                          self._epoch,
                                                                                                          self._iteration,
                                                                                                          acc,
                                                                                                          loss))
                print('################################################################################')
    
    def evaluate(self, split, log=True, confusion=False):
        
        self.model.eval()
        
        if confusion:
            conf_matrix = np.zeros((self.n_outputs, self.n_outputs))
        
        cumulative_loss = 0.
        cumulative_acc = 0
        n_samples = 0
        for batch_idx, (x_test, t_test) in enumerate(self._dataloaders[split]):
            x_test = x_test.to(self.device)
            t_test = t_test.to(self.device)
            
            y_test = self.model(x_test)
            
            if confusion:
                conf_matrix += compute_confusion_matrix(y_test.detach(), t_test, list(range(self.n_outputs)))
            
            # predictions.append(y_test.detach())
            # targets.append(t_test.detach())
            n_samples += x_test.shape[0]
            cumulative_acc += accuracy(y_test, t_test) * x_test.shape[0]
            cumulative_loss += self._loss_function(y_test, t_test).mean().item() * x_test.shape[0]
            
            del x_test
            del y_test
            del t_test
        
        acc = cumulative_acc / n_samples
        loss = cumulative_loss / n_samples
        
        if log:
            self.log(acc, loss, split)
        
        if confusion:
            return acc, loss, conf_matrix
        else:
            return acc, loss
    
    def plot(self):
        if self._visualization is not None:
            plot_exps.plot(self.logs, self.plotpath, self._show, outfig=self._visualization)
    
    def run(self):
        
        self._iteration = 0
        
        while self._epoch < self.epochs:
            starttime = datetime.datetime.now().timestamp()
            if self._time_limit is not None:
                if (datetime.datetime.now().timestamp() - self._start_time.timestamp()) / 60. > self._time_limit:
                    break
            
            if self._adapt_lr is not None and self._adapt_lr_type != "validation":
                self._adapt_lr()
            
            self.train()
            
            if self._eval_frequency < 0 and self._epoch % (-self._eval_frequency) == 0:
                self.valid()
            
            if self._plot_frequency < 0 and self._epoch % (-self._plot_frequency) == 0:
                self.plot()
            
            if self._backup_frequency < 0 and self._epoch % (-self._backup_frequency) == 0:
                self.backup()

            endtime = datetime.datetime.now().timestamp()
            
            if self._verbose > 1:
                duration = endtime - starttime
                print(f"Epoch {self._epoch} lasted {duration} seconds")

            self._epoch += 1
        
        if self._verbose > 1:
            print("###################################### Last Backup......... #######################################")
        
        self.backup()
        
        self.test()

    def _lr_scheduler_exponential_decay(self, verbose=False):
        #optimizer, epoch, epoch_start, init_lr, base_factor=.8, lr_decay_epoch=1, verbose=False):
        """
        Decay initial learning rate exponentially starting after epoch_start epochs
        The learning rate is multiplied with base_factor every lr_decay_epoch epochs
        """
        
        if self._lr_decay_schedule is not None:
            count = len([e for e in self._lr_decay_schedule if e <= self._epoch])
            lr = self._lr * (self._lr_decay_factor ** count)
        else:
            if self._epoch <= self._lr_decay_start:
                lr = self._lr
            else:
                lr = self._lr * (self._lr_decay_factor ** ((self._epoch - self._lr_decay_start) // self._lr_decay_epoch))
        if verbose:
            print('learning rate = {:6f}'.format(lr))
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
        return self._optimizer, lr
    
    def _lr_scheduler_valid_adaptive(self, verbose=False):
        """
        Decay initial learning rate exponentially by "_lr_decay_factor" starting after "_lr_decay_start" epochs
        The learning rate is multiplied with "_decay_factor" after the validation metric doesn't improve for "_lr_decay_epoch"
        """
        if self._epoch > self._lr_decay_start and self._epoch - max(self._last_adapt, self.best_valid_iteration) > 20:
            self._last_adapt = self._epoch
            self._lr_exp_steps += 1
        
        lr = self._lr * (self._lr_decay_factor ** self._lr_exp_steps)
        if verbose:
            print('learning rate = {:6f}'.format(lr))
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
        return self._optimizer, lr


def run_experiment(config):
    exp = Experiment(config)
    exp.run()
    
    utils.update_logs(exp.logs, utils.logs_path(config))
    utils.update_confusion(exp.conf_matrix, utils.logs_path(config))
    
    return exp.model, exp.logs


################################################################################
################################################################################


if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    parser = utils.args_exp_parameters(parser)
    
    config = parser.parse_args()
    
    # Train the model
    run_experiment(config)

import torch
import numpy as np
import pickle
import os
import torch.utils.data as data_utils
import torch.nn as nn
import torch.nn.functional as F

from models.monotonic_model import MonotonicModel
from models.fc_model import FCModel
from models.conv_model import ConvModel
from models.order_preserving_model import OrderPreservingModel
from models.model_with_temperature import ModelWithTemperature

import utils
from sklearn.model_selection import KFold

def acc(pred_logits, targets):
  preds = np.argmax(pred_logits, axis=1)
  return np.mean( preds == targets[:,0])

def torch_acc(pred_logits, targets):
  preds = torch.argmax(pred_logits, axis=1)
  return torch.mean(torch.eq( preds, targets).float())

def torch_topk(output, target, maxk=5):
  with torch.no_grad():
    batch_size = target.size(0)
    _, pred = output.topk(maxk,1,True,True)
    pred = pred.t()
    correct = pred.eq(target.view(1,-1).expand_as(pred))
    correct_k = correct[:maxk].view(-1).float().sum(0,keepdim=True)
    return correct_k.mul_(100.0/batch_size)

def get_dataloaders(dataset_config, is_training, verbose=True):
  """Creates a pytorch val, and test data loader given a dataset config.
  Args:
    - dataset_config: A dataset config. Should have the following fields:
      * root: (str) a path to the root of the dataset
      * name: (str) name of the dataset
      * batch_size: (int) size of batch
      * shuffle: (boolean) shuffle or not
      * num_workers: (int) number of workers
    - is_training: (bool). If true, splits the val dataset in to datset_config.num_folds sets
                   and and returns the list of corresponding train and test datasets
                   from the validation set.
                   If false, returns the original val and test datasets.
  Returns: val and test dataloaders, and number of classes
  """
  root = dataset_config.root
  name = dataset_config.name
  # load logits into memory
  pickle_path = os.path.join(root, name)
  assert(os.path.exists(pickle_path))
  with open(pickle_path, 'rb') as f:
    (y_logits_val, y_val), (y_logits_test, y_test) = pickle.load(f)

  if len(y_val.shape) == 1:
    y_val = y_val[...,None]

  if len(y_test.shape) == 1:
    y_test = y_test[...,None]

  # show some statistics
  if verbose:
    print("y_logits_val:", y_logits_val.shape)
    print("y_true_val:", y_val.shape)
    print("y_logits_test:", y_logits_test.shape)
    print("y_true_test:", y_test.shape)
    print("val accuracy: {}".format(acc(y_logits_val, y_val)))
    print("test accuracy: {}".format(acc(y_logits_test, y_test)))

  num_classes = y_logits_val.shape[1]
  tensorify = lambda x: torch.tensor(x)
  if is_training:
    assert('num_folds' in dataset_config)
    kf = KFold(n_splits=dataset_config.num_folds, shuffle=True, random_state=utils.get_rng())
    train_loaders = []
    test_loaders = []
    for train_indices, test_indices in kf.split(y_logits_val):
      X_train, X_test = y_logits_val[train_indices], y_logits_val[test_indices]
      Y_train, Y_test = y_val[train_indices], y_val[test_indices]
      [X_train, X_test, Y_train, Y_test] = map(tensorify, [X_train, X_test, Y_train, Y_test])
      train_dataset = data_utils.TensorDataset(X_train, Y_train)
      test_dataset = data_utils.TensorDataset(X_test, Y_test)
      train_loaders.append(data_utils.DataLoader(train_dataset, batch_size=dataset_config.batch_size,
                                                 shuffle=dataset_config.shuffle,
                                                 num_workers=dataset_config.num_workers))
      test_loaders.append(data_utils.DataLoader(test_dataset, batch_size=dataset_config.batch_size,
                                                shuffle=False,
                                                num_workers=1))
    return train_loaders, test_loaders, num_classes
  else:
    [y_logits_val, y_val, y_logits_test, y_test] = map(tensorify,
      [y_logits_val, y_val, y_logits_test, y_test])
    val_dataset = data_utils.TensorDataset(y_logits_val, y_val)
    test_dataset = data_utils.TensorDataset(y_logits_test, y_test)
    val_loader = data_utils.DataLoader(val_dataset, batch_size=dataset_config.batch_size,
                                       shuffle=dataset_config.shuffle,
                                       num_workers=dataset_config.num_workers)
    test_loader = data_utils.DataLoader(test_dataset, batch_size=dataset_config.batch_size,
                                       shuffle=False,
                                       num_workers=1)
    return [val_loader], [test_loader], num_classes

def get_order_preserving_model(model_config, num_classes):
  """Creates an order preserving model
    Args:
      - model_config: (EasyDict) a model config definition. It has different
                    fields based on the model. Some fields are:
        * name: (str) should be 'order_preserving'
        * type: (str) should be one of fc or conv which represents the type
                of base model
        * params: (dict) parameters of related order preserving model.
        * base_params: (dict) parameters of related base model.
      - num_classes: (int) number of classes
    Returns:
      a model object
  """
  assert model_config.name == 'order_preserving'
  model_config.base_params.num_classes = num_classes
  model_config.params.num_classes = num_classes
  if model_config.type == 'fc':
    base_model = FCModel(**model_config.base_params)
  elif model_config.type == 'conv':
    base_model = ConvModel(**model_config.base_params)
  else:
    raise ValueError("base model type {} is not defined".format(model_config.type))
  return OrderPreservingModel(base_model, **model_config.params)

def get_model(model_config, num_classes):
  """Creates a model based on the model_config.
  Args:
    - model_config: (EasyDict) a model config definition. It has different
                    fields based on the model. Some fields are:
      * name: (str) name of the model. Name can be one of following:
        - monotonic
        - unconstrained
        - temperature_scaling
        - order_preserving
      * params: (dict) parameters related to the model
    - num_classes: (int) number of classes
  Returns:
    a model object
  """
  model_name = model_config.name
  if model_name == 'monotonic':
    model_config.params.num_classes = num_classes
    model = MonotonicModel(**model_config.params)
  elif model_name == 'unconstrained':
    model_config.params.num_classes = num_classes
    model = FCModel(**model_config.params)
  elif model_name == 'temperature_scaling':
    model = ModelWithTemperature()
  elif model_name == 'order_preserving':
    model = get_order_preserving_model(model_config, num_classes)
  else:
    raise ValueError("model_name {} is not defined".format(model_name))
  if "init" in model_config:
    init = model_config.init
    if init == 'msodir_init':
      dev = model.fc.weight.device
      with torch.no_grad():
        model.fc.weight.data = torch.eye(num_classes, device=dev)
        if model.fc.bias is not None:
          model.fc.bias.data = torch.zeros_like(model.fc.bias)
  return model

def get_optimizer(opt_config, model):
  """Creates an optimizer based on the opt_config and model parameters.
  Args:
    - opt_config: (dict) optimizer config. It should have the following fields:
      * name: (str) name of the optimizer. It should be one of SGD, Adam, or LBFGS
      * params: (dict) optimizer specific parameters
  Returns:
    - optimizer object
  """
  opt_name = opt_config.name
  if opt_name == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), **opt_config.params)
  elif opt_name == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), **opt_config.params)
  elif opt_name == 'LBFGS':
    optimizer = torch.optim.LBFGS(model.parameters(), **opt_config.params)
  else:
    raise ValueError("Optimizer {} is not defined".format(opt_name))
  return optimizer

def get_loss_fn(loss_config, model):
  """Creates a loss function based on loss_config.
  Args:
    loss_config: (dict) loss config with following parameters:
      - name: (str) name of the loss
      - params: (dict) loss parameters if any
    model: a model which its parameters might be used for defining
           custom regularized loss function like msodir.
  Returns:
    loss function (usually a nn.Module instance or a callable)
  """
  if loss_config.name == 'cross_entropy':
    loss_fn = nn.CrossEntropyLoss()
  elif loss_config.name == 'ece':
    #loss for ece should have this structure:
    #loss {
    #     name: "ece"
    #     params {
    #       nbins: 15
    #     }
    #}
    loss_fn = utils._ECELoss(**loss_config.params)
  elif loss_config.name == 'msodir':
    loss_fn = utils._MSODIRLoss(model, **loss_config.params)
  return loss_fn

def get_metrics_fn(nbins,**kwargs):
  metrics = {'nll': nn.CrossEntropyLoss(),
             'ece': utils._ECELoss(nbins,**kwargs),
             'acc': torch_acc,
             'brier': utils._BrierLoss(),
             'CwECE': utils._CwECELoss(nbins),
             'CwECEsum': utils._CwECELoss(nbins-1, avg=False), #This is used to have the
                                                               # same results as Kull etal.
             'top5': torch_topk,
             }
  return metrics

def get_metrics_op():
  ops = {'nll': min,
         'ece': min,
         'acc': max,
         'brier': min,
         'CwECE': min,
         'CwECEsum': min,
         'loss': min,
         'top5': max}
  return ops


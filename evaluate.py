"""Evaluates the model"""

import argparse
import logging
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import utils
import json
from easydict import EasyDict
from time import time
import pickle
import builder

parser = argparse.ArgumentParser()
parser.add_argument('--exp_dir', default='experiments/base_model', help="Directory containing config.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --exp_dir \
                     containing weights to load")
parser.add_argument('--save_logits', default=False, type=bool, help='save logits?')
parser.add_argument('--save_bins', default=False, type=bool, help='save ece bins?')
parser.add_argument('--save_bins_metric', default=None, type=str, help='which metric to save ece bins?')

def evaluate(model, loss_fn, dataloader, metrics, config, save_logits=False, exp_dir=None, save_prefix=None):
  """Evaluate the model on `num_steps` batches.

  Args:
      model: (torch.nn.Module) the neural network
      loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
      dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
      metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
      config: (config) hyperparameters
  """

  # set model to evaluation mode
  if isinstance(model, list):
    [m.eval() for m in model]
  else:
    model.eval()

  # summary for current eval loop
  summ = []
  all_logits = []
  all_scores = []
  all_labels = []
  # compute metrics over the dataset

  for data_batch, labels_batch in dataloader:
    # move to GPU if available
    if config.cuda:
        data_batch, labels_batch = data_batch.cuda(async=True), labels_batch.cuda(async=True)
    # fetch the next evaluation batch
    data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)

    # compute model output
    if isinstance(model, list):
      output_batch = torch.mean(torch.stack([m(data_batch) for m in model]), dim=0)
    else:
      output_batch = model(data_batch)

    all_logits.append(data_batch)
    all_scores.append(output_batch)
    all_labels.append(labels_batch)

  logits = torch.cat(all_logits)
  scores = torch.cat(all_scores)
  labels = torch.cat(all_labels).type(torch.int64)

  pre_metrics = {metric: float(metrics[metric](logits, labels[:,0])) for metric in metrics}
  post_metrics = {metric: float(metrics[metric](scores, labels[:,0])) for metric in metrics}
  pre_metrics.update({'loss': float(loss_fn(logits, labels[:,0].long()))})
  post_metrics.update({'loss': float(loss_fn(scores, labels[:,0].long()))})

  # compute mean of all metrics in summary
  metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in pre_metrics.items())
  logging.info("- Eval metrics pre-calibration: " + metrics_string)
  metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in post_metrics.items())
  logging.info("- Eval metrics post-calibration: " + metrics_string)

  if save_logits and exp_dir is not None:
    prefix = ''
    if save_prefix is not None:
      prefix = save_prefix + '_'
    np.save(os.path.join(exp_dir, prefix+'logits.npy'), logits.detach().cpu().numpy())
    np.save(os.path.join(exp_dir, prefix+'scores.npy'), scores.detach().cpu().numpy())
    np.save(os.path.join(exp_dir, prefix+'labels.npy'), labels.detach().cpu().numpy())

  return pre_metrics, post_metrics

if __name__ == '__main__':
  """
      Evaluate the model on the test set.
  """
  # Load the parameters
  args = parser.parse_args()
  json_path = os.path.join(args.exp_dir, 'config.json')
  assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
  with open(json_path, 'r') as f:
    config = EasyDict(json.load(f))

  # use GPU if available
  config.cuda = torch.cuda.is_available()     # use GPU is available

  # Set the random seed for reproducible experiments
  seed = 1357 if 'seed' not in config else config.seed
  torch.manual_seed(1357)
  if config.cuda: torch.cuda.manual_seed(1357)
  utils.fix_rng_seed(seed)

  # Get the logger
  utils.set_logger(os.path.join(args.exp_dir, 'evaluate.log'))

  # Create the input data pipeline
  logging.info("Creating the dataset...")

  # fetch dataloaders
  val_loaders, test_loaders, num_classes = builder.get_dataloaders(config.dataset, is_training=False, verbose=config.verbose)
  config.num_classes = num_classes

  logging.info("- done.")

  val_loader = val_loaders[0]
  test_loader = test_loaders[0]
  metrics_dict= builder.get_metrics_op()
  metrics_dict.pop('loss')
  metric_names = metrics_dict.keys()

  for metric_name in metric_names:
    all_pre_metrics= []
    all_post_metrics= []
    for i in range(config.dataset.num_folds):

      # Define the model
      model = builder.get_model(config.model, num_classes)

      model = model.cuda() if config.cuda else model

      loss_fn = builder.get_loss_fn(config.loss, model)
      kwargs = {}
      if args.save_bins:
        assert(args.save_bins_metric is not None)
        if metric_name == args.save_bins_metric:
          kwargs['save_bins'] = True
          kwargs['save_path'] = os.path.join(args.exp_dir, 'fold_{}'.format(i+1), 'bins.json')
      metrics = builder.get_metrics_fn(config.nbins, **kwargs)
      exp_dir = os.path.join(args.exp_dir, 'fold_{}'.format(i+1))
      assert(os.path.exists(exp_dir))

      logging.info("Starting evaluation on fold {}".format(i+1))

      # Reload weights from the saved file
      utils.load_checkpoint(os.path.join(exp_dir, args.restore_file + '_{}.pth.tar'.format(metric_name)), model)

      # Evaluate
      save_logits = args.save_logits if metric_name == 'nll' else False
      pre_metrics, post_metrics = evaluate(model, loss_fn,
                                           test_loader, metrics, config, save_logits, exp_dir)
      save_path = os.path.join(exp_dir, "post_metrics_test_{}_{}.json".format(args.restore_file, metric_name))
      utils.save_dict_to_json(post_metrics, save_path)
      save_path = os.path.join(exp_dir, "pre_metrics_test_{}_{}.json".format(args.restore_file, metric_name))
      utils.save_dict_to_json(pre_metrics, save_path)
      all_pre_metrics.append(pre_metrics)
      all_post_metrics.append(post_metrics)

    final_pre_metrics = {metric: np.mean([m[metric] for m in all_pre_metrics]) for metric in all_pre_metrics[0]}
    final_post_metrics = {metric: np.mean([m[metric] for m in all_post_metrics]) for metric in all_post_metrics[0]}
    pre_metrics_path = os.path.join(args.exp_dir, 'cross_val_test_pre_metrics_{}_{}.json'.format(args.restore_file, metric_name))
    post_metrics_path = os.path.join(args.exp_dir, 'cross_val_test_post_metrics_{}_{}.json'.format(args.restore_file, metric_name))
    utils.save_dict_to_json(final_pre_metrics, pre_metrics_path)
    utils.save_dict_to_json(final_post_metrics, post_metrics_path)

    #ensemble (This is used following Kull etal.)
    #          It improves their results substantially,
    #          However, for our case, the improvement
    #          is marginal.
    logging.info("Evaluating in ensemble mode...")
    val_loaders, test_loaders, num_classes = builder.get_dataloaders(config.dataset, is_training=False, verbose=config.verbose)
    models = []
    kwargs = {}
    if args.save_bins:
      assert(args.save_bins_metric is not None)
      if metric_name == args.save_bins_metric:
        kwargs['save_bins'] = True
        kwargs['save_path'] = os.path.join(args.exp_dir, 'fold_{}'.format(i+1), 'bins.json')
    metrics = builder.get_metrics_fn(config.nbins, **kwargs)
    exp_dir = os.path.join(args.exp_dir, 'fold_{}'.format(i+1))
    assert(os.path.exists(exp_dir))
    val_loader = val_loaders[0]
    test_loader = test_loaders[0]
    for i in range(config.dataset.num_folds):
      # Define the model
      model = builder.get_model(config.model, num_classes)
      loss_fn = builder.get_loss_fn(config.loss, model)
      model = model.cuda() if config.cuda else model

      # Reload weights from the saved file
      utils.load_checkpoint(os.path.join(exp_dir, args.restore_file + '_{}.pth.tar'.format(metric_name)), model)
      models.append(model)

      # Evaluate
    save_logits = args.save_logits
    exp_dir = os.path.join(args.exp_dir, 'ensemble')
    if not os.path.exists(exp_dir):
      os.makedirs(exp_dir)
    pre_metrics, post_metrics = evaluate(model, loss_fn,
                                         test_loader, metrics, config, save_logits, exp_dir, save_prefix=metric_name)
    save_path = os.path.join(exp_dir, "post_metrics_test_ensemble_{}_{}.json".format(args.restore_file, metric_name))
    utils.save_dict_to_json(post_metrics, save_path)


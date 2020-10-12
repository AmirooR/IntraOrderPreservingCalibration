"""Calibrate the model"""

import argparse
import logging
import os

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import utils
from evaluate import evaluate
from easydict import EasyDict
import json
import builder
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--exp_dir', default='exp_dir/base', help="Directory containing config.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --exp_dir containing weights \
                          to reload before training")  # 'best' or 'train'
parser.add_argument('--singlefold', type=bool, default=False, help='Uses cross validation if False, \
                          otherwise, uses full validation data for training and full test data \
                          for evaluation during training')

def calibrate(model, optimizer, loss_fn, dataloader, metrics, config):
    """Calibrate the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        config: (config) hyperparameters
        num_steps: (int) number of batches to train on, each of size config.batch_size
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            # move to GPU if available
            if config.cuda:
                train_batch, labels_batch = train_batch.cuda(async=True), labels_batch.cuda(async=True)
            # convert to torch Variables
            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

            # compute model output and loss (outputs are logits)
            output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch[:,0].long())

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()
            loss_val = loss.data.cpu().numpy()

            # Evaluate summaries only once in a while
            if (i+1) % config.calibrate.save_summary_steps == 0:
                # compute all metrics on this batch
                summary_batch = {metric:float(metrics[metric](output_batch, labels_batch[:,0].long()))
                                 for metric in metrics}
                #loss_val = loss.data.cpu().numpy()
                summary_batch['loss'] = loss_val
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss_val)

            t.set_postfix(loss='{:05.5f}'.format(loss_avg()))
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.5f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)
    return metrics_mean


def calibrate_fullgrad(model, optimizer, loss_fn, dataloader, metrics, config, val_dataloader):
    """Calibrate the model on full data as batch. Useful when optimizer is LBFGS.

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        config: (config) hyperparameters
        num_steps: (int) number of batches to train on, each of size config.batch_size
    """

    # set model to training mode
    model.train()

    logits_list = []
    labels_list = []
    val_logits_list = []
    val_labels_list = []
    # Use tqdm for progress bar
    with torch.no_grad():
      with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            # move to GPU if available
            if config.cuda:
                train_batch, labels_batch = train_batch.cuda(async=True), labels_batch.cuda(async=True)
            logits_list.append(train_batch)
            labels_list.append(labels_batch)
            t.set_postfix(i='{}'.format(i))
            t.update()
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)
    if False:
      with torch.no_grad():
        with tqdm(total=len(val_dataloader)) as t:
          for i, (train_batch, labels_batch) in enumerate(val_dataloader):
              # move to GPU if available
              if config.cuda:
                  train_batch, labels_batch = train_batch.cuda(async=True), labels_batch.cuda(async=True)
              val_logits_list.append(train_batch)
              val_labels_list.append(labels_batch)
              t.set_postfix(i='{}'.format(i))
              t.update()
          val_logits = torch.cat(val_logits_list)
          val_labels = torch.cat(val_labels_list)

    pre_metrics = {metric: metrics[metric](logits,labels[:,0].long()).item() for metric in metrics}
    logging.info("Before calibration")
    for metric in pre_metrics:
      logging.info(" - {}: {}".format(metric, pre_metrics[metric]))
    i = 0
    def eval():
      global i
      i += 1
      calibrated_logits = model(logits)
      loss = loss_fn(calibrated_logits, labels[:,0].long())
      print("LOSS: {}".format(loss))
      loss.backward()
      print("ITERATION: {}".format(i))
      return loss

    optimizer.zero_grad()
    optimizer.step(eval)
    logging.info("After calibration")
    post_metrics = {metric: metrics[metric](model(logits), labels[:,0].long()).item() for metric in metrics}
    for metric in post_metrics:
      logging.info(" - {}: {}".format(metric, post_metrics[metric]))

    return post_metrics


def calibrate_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, metrics, config, exp_dir,
                           restore_file=None):
  """Calibrate the model and evaluate every epoch.

  Args:
      model: (torch.nn.Module) the neural network
      train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
      val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
      optimizer: (torch.optim) optimizer for parameters of model
      loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
      metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
      config: (config) hyperparameters
      exp_dir: (string) directory containing config, weights and log
      sampler: (Sampler) the sampler which used for sampling data.
      restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
  """
  # reload weights from restore_file if specified
  if restore_file is not None:
      restore_path = os.path.join(exp_dir, restore_file + '.pth.tar')
      if os.path.exists(restore_path):
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)
      else:
        logging.info("WARNING: restore file {} does not exists".format(restore_path))

  metric_ops = builder.get_metrics_op()
  fn_to_val = {min: np.inf, max: -np.inf}
  best_metric_vals = {metric:fn_to_val[ metric_ops[metric]] for metric in metric_ops}
  best_metrics = {}

  if 'early_stopping' in config.calibrate:
    early_stopping = utils.EarlyStopping(patience=config.calibrate.early_stopping.patience,
                                         delta=0)

  all_metrics = defaultdict(list)
  all_train_metrics = defaultdict(list)

  for epoch in range(config.calibrate.num_epochs):
    # Run one epoch
    logging.info("Epoch {}/{}".format(epoch + 1, config.calibrate.num_epochs))

    # compute number of batches in one epoch (one full pass over the training set)
    if config.optimizer.name == "LBFGS":
      train_metrics = calibrate_fullgrad(model, optimizer, loss_fn, train_dataloader, metrics, config, val_dataloader)
    else:
      train_metrics = calibrate(model, optimizer, loss_fn, train_dataloader, metrics, config)
    for metric in train_metrics:
      all_train_metrics[metric].append(train_metrics[metric])

    # Evaluate for one epoch on validation set
    pre_metrics, post_metrics = evaluate(model, loss_fn, val_dataloader, metrics, config)

    for metric in metrics:
      all_metrics[metric].append(post_metrics[metric])
      if metric_ops[metric] == min:
        is_best = post_metrics[metric] <= best_metric_vals[metric]
      else:
        is_best = post_metrics[metric] >= best_metric_vals[metric]

      # Save weights
      utils.save_checkpoint({'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'optim_dict' : optimizer.state_dict()},
                            is_best=is_best,
                            checkpoint=exp_dir,
                            name=metric)

      # If best_eval, best_save_path
      post_metrics.update({'epoch':epoch+1})
      if is_best:
        logging.info("- Found new best {}".format(metric))
        best_metric_vals[metric] = post_metrics[metric]
        best_metrics[metric] = post_metrics

        # Save best val metrics in a json file in the model directory
        best_json_path = os.path.join(exp_dir, "metrics_val_best_{}.json".format(metric))
        utils.save_dict_to_json(post_metrics, best_json_path)

    # Save latest val metrics in a json file in the model directory

    last_json_path = os.path.join(exp_dir, "metrics_val_last_weights.json")
    utils.save_dict_to_json(post_metrics, last_json_path)
    if 'early_stopping' in config.calibrate:
      early_stopping(post_metrics['loss'], model)
      if early_stopping.early_stop:
        logging.info("EARLY STOPPING")
        break
  utils.save_dict_to_json(all_metrics, os.path.join(exp_dir, "all_metrics_eval.json"))
  utils.save_dict_to_json(all_train_metrics, os.path.join(exp_dir, "all_metrics_train.json"))
  return best_metrics

if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.exp_dir, 'config.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    with open(json_path, 'r') as f:
      config = EasyDict(json.load(f))

    # use GPU if available
    config.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    seed = 1357 if 'seed' not in config else config.seed
    torch.manual_seed(seed)
    if config.cuda: torch.cuda.manual_seed(seed)
    utils.fix_rng_seed(seed)

    # Set the logger
    utils.set_logger(os.path.join(args.exp_dir, 'calibrate.log'))
    logging.info(args.exp_dir)

    # Create the input data pipeline
    logging.info("Loading the dataset...")
    is_training = not args.singlefold
    logging.info("is_trainig: {}".format(is_training))

    # fetch dataloaders
    val_loaders, test_loaders, num_classes = builder.get_dataloaders(config.dataset, is_training=is_training, verbose=config.verbose)
    config.num_classes = num_classes

    logging.info("- done.")
    all_metrics = []

    for i, (val_loader, test_loader) in enumerate(zip(val_loaders, test_loaders)):
      # Define the model and optimizer
      model = builder.get_model(config.model, num_classes)

      model = model.cuda() if config.cuda else model

      #get the optimizer
      optimizer = builder.get_optimizer(config.optimizer, model)

      # fetch loss function and metrics
      loss_fn = builder.get_loss_fn(config.loss, model)
      metrics = builder.get_metrics_fn(config.nbins)
      exp_dir = os.path.join(args.exp_dir, 'fold_{}'.format(i+1))
      if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

      # Train the model
      logging.info("Starting calibration for {} epoch(s) on fold {}".format(config.calibrate.num_epochs, i+1))
      best_metrics = calibrate_and_evaluate(model, val_loader, test_loader,
                                            optimizer, loss_fn, metrics,
                                            config, exp_dir, args.restore_file)
      all_metrics.append(best_metrics)

    logging.info("Final Mean Metrics")
    final_metrics = {metric: np.mean([m[metric][metric] for m in all_metrics]) for metric in metrics}
    final_metrics_path = os.path.join(args.exp_dir, 'cross_val_metrics.json')
    utils.save_dict_to_json(final_metrics, final_metrics_path)

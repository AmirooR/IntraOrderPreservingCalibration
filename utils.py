import json
import logging
import os
import shutil
import numpy as np
import torch
from datetime import datetime, timedelta
from torch import nn, optim
from torch.nn import functional as F
from models.fc_model import FCModel
from sklearn.preprocessing import label_binarize

_RNG_SEED = None

def split(a, n):
  k, m = divmod(len(a), n)
  return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def fix_rng_seed(seed):
    """
    Call this function at the beginning of program to fix rng seed.
    Args:
        seed (int):
    Note:
        See https://github.com/tensorpack/tensorpack/issues/196.
    Example:
        Fix random seed in both tensorpack and tensorflow.
    .. code-block:: python
            import utils
            seed = 42
            utils.fix_rng_seed(seed)
            torch.manual_seed(seed)
            if config.cuda: torch.cuda.manual_seed(seed)
            # run trainer
    """
    global _RNG_SEED
    _RNG_SEED = int(seed)


def get_rng(obj=None):
    """
    Get a good RNG seeded with time, pid and the object.
    Args:
        obj: some object to use to generate random seed.
    Returns:
        np.random.RandomState: the RNG.
    """
    seed = (id(obj) + os.getpid() +
            int(datetime.now().strftime("%Y%m%d%H%M%S%f"))) % 4294967295
    if _RNG_SEED is not None:
        seed = _RNG_SEED
    return np.random.RandomState(seed)

class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total/float(self.steps)


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    def to_float(d):
      for k,v in d.items():
        if type(v) is dict:
          d[k] = to_float(v)
        elif isinstance(v, np.float32) or isinstance(v, np.float64):
          d[k] = float(v)
        elif isinstance(v, list):
          d[k] = [float(x) for x in v]
      return d

    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = to_float(d) #{k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=2)

def save_checkpoint(state, is_best, checkpoint, name=None):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    #else:
    #print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
      save_name = 'best.pth.tar' if name is None else 'best_{}.pth.tar'.format(name)
      shutil.copyfile(filepath, os.path.join(checkpoint, save_name))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise ValueError("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint

class _ECELoss(nn.Module):
  """
  Calculates the Expected Calibration Error of a model.

  The input to this loss is the logits of a model, NOT the softmax scores.

  This divides the confidence outputs into equally-sized interval bins.
  In each bin, we compute the confidence gap:

  bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

  We then return a weighted average of the gaps, based on the number
  of samples in each bin

  See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
  "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
  2015.
  """
  def __init__(self, n_bins=15, save_bins=False, save_path=None):
    """
    n_bins (int): number of confidence interval bins
    """
    super(_ECELoss, self).__init__()
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    self.bin_lowers = bin_boundaries[:-1]
    self.bin_uppers = bin_boundaries[1:]
    self.save_bins = save_bins
    self.save_path = save_path

  def forward(self, logits, labels):
    softmaxes = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(labels)
    ece = torch.zeros(1, device=logits.device)
    if self.save_bins:
      bin_data = {'bin_lowers': [], 'bin_uppers': [], 'props': [], 'accs': [], 'confs': []}
    for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
      # Calculated |confidence - accuracy| in each bin
      in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
      prop_in_bin = in_bin.float().mean()
      if prop_in_bin.item() > 0:
        accuracy_in_bin = accuracies[in_bin].float().mean()
        avg_confidence_in_bin = confidences[in_bin].mean()
        ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        if self.save_bins:
          bin_data['bin_lowers'].append(bin_lower.item())
          bin_data['bin_uppers'].append(bin_upper.item())
          bin_data['props'].append(prop_in_bin.item())
          bin_data['accs'].append(accuracy_in_bin.item())
          bin_data['confs'].append(avg_confidence_in_bin.item())
    if self.save_bins:
      save_dict_to_json(bin_data, self.save_path)
    return ece

class _CwECELoss(nn.Module):
  """
  Calculates the Class-wise Expected Calibration Error of a model.

  The input to this loss is the logits of a model, NOT the softmax scores.

  This divides the confidence outputs of each class j into equally-sized
  interval bins. In each bin, we compute the confidence gap:

  bin_gap = | avg_confidence_in_bin_j - accuracy_in_bin_j |

  We then return a weighted average of the gaps, based on the number
  of samples in each bin
  """
  def __init__(self, n_bins=15, avg=True):
    """
    n_bins (int): number of confidence interval bins
    """
    super(_CwECELoss, self).__init__()
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    self.bin_lowers = bin_boundaries[:-1]
    self.bin_uppers = bin_boundaries[1:]
    self.avg = avg

  def forward(self, logits, labels):
    softmaxes = F.softmax(logits, dim=1)
    num_classes = logits.shape[1]
    cw_ece = torch.zeros(1, device=logits.device)
    for j in range(num_classes):
      confidences_j = softmaxes[:,j]
      ece_j = torch.zeros(1, device=logits.device)
      for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
        in_bin = confidences_j.gt(bin_lower.item()) * confidences_j.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
          accuracy_j_in_bin = labels[in_bin].eq(j).float().mean()
          avg_confidence_j_in_bin = confidences_j[in_bin].mean()
          ece_j += torch.abs(avg_confidence_j_in_bin - accuracy_j_in_bin) * prop_in_bin
      cw_ece += ece_j
    if self.avg:
      return cw_ece/num_classes
    else:
      return cw_ece


# The next two functions are copied from Kull etal implementation
# for testing
def binary_ECE(probs, y_true, power = 1, bins = 15):

    idx = np.digitize(probs, np.linspace(0, 1, bins)) - 1
    bin_func = lambda p, y, idx: (np.abs(np.mean(p[idx]) - np.mean(y[idx])) ** power) * np.sum(idx) / len(probs)

    ece = 0
    for i in np.unique(idx):
        ece += bin_func(probs, y_true, idx == i)
    return ece

def classwise_ECE(probs, y_true, power = 1, bins = 15):

    probs = np.array(probs)
    if not np.array_equal(probs.shape, y_true.shape):
        y_true = label_binarize(np.array(y_true), classes=range(probs.shape[1]))

    n_classes = probs.shape[1]

    return np.sum(
        [
            binary_ECE(
                probs[:, c], y_true[:, c].astype(float), power = power, bins = bins
            ) for c in range(n_classes)
        ]
    )

class _CwECELossDir(nn.Module):
  """
  Calculates the Class-wise Expected Calibration Error of a model.

  The input to this loss is the logits of a model, NOT the softmax scores.

  This divides the confidence outputs of each class j into equally-sized
  interval bins. In each bin, we compute the confidence gap:

  bin_gap = | avg_confidence_in_bin_j - accuracy_in_bin_j |

  We then return a weighted average of the gaps, based on the number
  of samples in each bin
  """
  def __init__(self, n_bins=15):
    """
    n_bins (int): number of confidence interval bins
    """
    super(_CwECELossDir, self).__init__()
    self.n_bins = n_bins

  def forward(self, logits, labels):
    probs = F.softmax(logits, dim=1).detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy()
    cwece = classwise_ECE(probs, y_true, bins=self.n_bins)
    return torch.tensor(cwece, device=logits.device, dtype=logits.dtype)

class _MCELoss(nn.Module):
  """
  Calculates the Maximum Calibration Error of a model.

  The input to this loss is the logits of a model, NOT the softmax scores.

  This divides the confidence outputs into equally-sized interval bins.
  In each bin, we compute the confidence gap:

  bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

  We then return a maximum of the gaps

  See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
  "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
  2015.
  """
  def __init__(self, n_bins=15):
    """
    n_bins (int): number of confidence interval bins
    """
    super(_MCELoss, self).__init__()
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    self.bin_lowers = bin_boundaries[:-1]
    self.bin_uppers = bin_boundaries[1:]

  def forward(self, logits, labels):
    softmaxes = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(labels)
    cal_errors = []
    for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
      # Calculated |confidence - accuracy| in each bin
      in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
      prop_in_bin = in_bin.float().mean()
      if prop_in_bin.item() > 0:
        accuracy_in_bin = accuracies[in_bin].float().mean()
        avg_confidence_in_bin = confidences[in_bin].mean()
        cal_errors.append(torch.abs(avg_confidence_in_bin - accuracy_in_bin))
    return torch.max(torch.stack(cal_errors))

class _BrierLoss(nn.Module):
  """
  Calculates the Brier Error of a model.

  The input to this loss is the logits of a model, NOT the softmax scores.

  We then return a mean square of the gaps between one-hot labels and the
  predicted scores.
  """
  def __init__(self):
    """
    """
    super(_BrierLoss, self).__init__()

  def forward(self, logits, labels):
    softmaxes = F.softmax(logits, dim=1)
    labels_onehot = torch.zeros_like(softmaxes)
    labels_onehot.scatter_(1, labels[...,None], 1)
    diff = (labels_onehot - softmaxes)
    diff = diff *diff
    return diff.mean()

class _MSODIRLoss(object):
  """
  Calculates the nll + Matrix scaling off-diagonal andbias regularization
  of a model.

  The input to this loss is the logits of a model, NOT the softmax scores.

  NOTE: This loss only works with FCModel with zero hidden
        layers (i.e., num_hiddens is [])
  """
  def __init__(self, model, weight_lambda=5e-4, bias_mu=5e-4):
    """
    Args:
      model: (models.fc_model.FCModel object)
      weight_lambda: regularization weight for weights of fc in model
      bias_mu: regularization weights for bias of fc in model
    """
    super(_MSODIRLoss, self).__init__()
    assert(isinstance(model, FCModel))
    assert(len(model.num_hiddens) == 0)
    self.model = model
    self.ce = nn.CrossEntropyLoss()
    self.weight_lambda = weight_lambda
    self.bias_mu = bias_mu

  def __call__(self, logits, labels):
    ce_part = self.ce(logits, labels)
    weight_part =  (1 -
                    torch.eye(self.model.fc.weight.shape[0])
                    ).to(logits.device)*self.model.fc.weight
    weight_part = torch.sum(weight_part*weight_part)
    if self.model.fc.bias is not None:
      bias_part = torch.sum(self.model.fc.bias * self.model.fc.bias)
    else:
      bias_part=0
    return ce_part + self.weight_lambda * weight_part + self.bias_mu * bias_part


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
          print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(self.val_loss_min, self.val_loss_min))
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

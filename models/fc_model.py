import torch
import torch.nn as nn
import torch.nn.functional as f


class FCModel(nn.Module):
  def __init__(self, num_hiddens, residual=False, num_classes=10,
                     batch_norm=False, log_softmax=False):
    super(FCModel, self).__init__()
    self._residual = residual
    self.num_hiddens = num_hiddens #it is used in msodir_loss
    self.num_classes = num_classes #it is used in msodir_loss
    self.log_softmax = log_softmax
    last_hidden = num_classes
    calib_layers = []
    for num_hidden in num_hiddens:
      if batch_norm:
        calib_layers.append(nn.BatchNorm1d(last_hidden))
      calib_layers.append(nn.Linear(last_hidden, num_hidden, bias=True))
      calib_layers.append(nn.ReLU())
      last_hidden = num_hidden
    if len(calib_layers) > 0:
      self.calib_layers = nn.Sequential(*calib_layers)
    else:
      self.calib_layers = lambda x: x
    if batch_norm:
      self.bn = nn.BatchNorm1d(num_hiddens[-1])
    else:
      self.bn = lambda x: x
    if len(num_hiddens) > 0:
      self.fc = nn.Linear(num_hiddens[-1], num_classes)
    else:
      self.fc = nn.Linear(num_classes, num_classes)

  def forward(self, logits):
    if self.log_softmax:
      logits = f.log_softmax(logits, dim=1)
    out = self.calib_layers(logits)
    out = self.fc(self.bn(out))
    if self._residual:
      out = out + logits
    return out


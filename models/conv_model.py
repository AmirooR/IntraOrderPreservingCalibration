import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvModel(nn.Module):
  def __init__(self, num_hiddens, kernel_size_or_sizes=3,
                     residual=False, num_classes=10, log_softmax=False):
    super(ConvModel, self).__init__()
    kernel_sizes = kernel_size_or_sizes if isinstance(kernel_size_or_sizes, list) \
                                        else [kernel_size_or_sizes]*(len(num_hiddens)+1)

    assert(len(kernel_sizes) == len(num_hiddens)+1)
    self.kernel_sizes = kernel_sizes
    self._residual = residual
    self.num_hiddens = num_hiddens
    self.num_classes = num_classes
    self.log_softmax = log_softmax
    in_channels = 1
    conv_layers = nn.ModuleList([])
    for num_hidden, k_size in zip(num_hiddens, self.kernel_sizes):
      conv_layers.append(nn.Conv1d(in_channels, num_hidden, k_size, bias=True))
      in_channels = num_hidden
    conv_layers.append(nn.Conv1d(in_channels, 1, kernel_sizes[-1], bias=True))
    self.conv_layers = conv_layers

  def forward(self, logits):
    if self.log_softmax:
      logits = f.log_softmax(logits, dim=1)
    #[B,1,NUM_CLASSES]
    h = logits[:,None,:]
    for k_size, conv_layer in zip(self.kernel_sizes, self.conv_layers):
      pad_size = k_size//2
      h = F.pad(h, (pad_size, pad_size), mode='replicate')
      h = conv_layer(h)
      if conv_layer != self.conv_layers[-1]:
        h = F.relu(h)
    if self._residual:
      h = h + logits[:,None,:]
    return h[:,0,:]


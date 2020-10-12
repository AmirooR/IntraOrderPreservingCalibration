import torch
import torch.nn as nn
import torch.nn.functional as f


class ModelWithTemperature(nn.Module):
  def __init__(self):
    super(ModelWithTemperature, self).__init__()
    self.temperature = nn.Parameter(torch.ones(1)*1.5)

  def forward(self, logits):
    return logits / self.temperature


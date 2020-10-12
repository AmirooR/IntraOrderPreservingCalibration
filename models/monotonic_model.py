import torch
import torch.nn as nn
import torch.nn.functional as f
from models.UMNN import MonotonicNN

class MonotonicModel(nn.Module):
  def __init__(self, num_hiddens, nb_steps, device, num_classes=10, conditioned=False,
               subsample_ratio=1.,
               add_condition_to_integrand=False):
    super(MonotonicModel, self).__init__()
    self.conditioned = conditioned
    self.subsample_ratio = subsample_ratio
    assert(subsample_ratio > 0 and subsample_ratio <= 1.)
    if conditioned:
      in_d = 1 + num_classes
    else:
      in_d = 2
    self.model_monotonic = MonotonicNN(in_d, num_hiddens, nb_steps=nb_steps, dev=device,
                                       add_condition_to_integrand=add_condition_to_integrand).to(device)

  def forward(self, logits):
    # logits: [batch_size, num_classes]
    # flat_out: [batch_size*num_classes,1]
    flat_out = logits.contiguous().view(-1).unsqueeze(1)
    if self.conditioned:
      # h: [batch_size*num_classes, num_classes]
      h = logits.repeat_interleave(logits.shape[1], dim=0)
    else:
      h = torch.zeros_like(flat_out)
    mono_out = self.model_monotonic(flat_out, h).contiguous().view(logits.shape)
    return mono_out


import torch
import torch.nn as nn
import torch.nn.functional as F


class OrderPreservingModel(nn.Module):
  def __init__(self, base_model, invariant=False, residual=False,
               num_classes=10, m_activation=F.softplus):
    super(OrderPreservingModel, self).__init__()
    self.base_model = base_model
    self.num_classes = num_classes #it is used in msodir_loss
    self.invariant = invariant
    self.m_activation = m_activation
    self.residual = residual

  def compute_u(self, sorted_logits):
    diffs = sorted_logits[:,:-1] - sorted_logits[:,1:]
    diffs = torch.cat((diffs, torch.ones((diffs.shape[0],1),
                                          dtype=diffs.dtype,
                                          device=diffs.device)), dim=1)
    assert(torch.all(diffs >= 0)), 'diffs should be positive: {}'.format(diffs)
    return diffs.flip([1])


  def forward(self, logits):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    _, unsorted_indices = torch.sort(sorted_indices, descending=False)
    #[B, C]
    u = self.compute_u(sorted_logits)
    inp = sorted_logits if self.invariant else logits
    m = self.base_model(inp)
    m[:,1:] = self.m_activation(m[:,1:].clone())
    m[:,0] = 0
    um = torch.cumsum(u*m,1).flip([1])
    out = torch.gather(um,1,unsorted_indices)
    if self.residual:
      out = out + logits
    return out


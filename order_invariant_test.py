# coding: utf-8
import torch


from models.order_preserving_model import OrderPreservingModel
from models.fc_model import FCModel
from models.conv_model import ConvModel

fc_model = FCModel([15])
fc_op_model = OrderPreservingModel(fc_model, invariant=True)
logits = torch.randn(2,10)
out = fc_op_model(logits)
_,ind1 = torch.sort(logits)
_,ind2 = torch.sort(out)
assert(torch.all(ind1 == ind2))
l = logits.clone().detach()
logits[:, 2] = l[:, 5]
logits[:, 5] = l[:, 2]
out3 = fc_op_model(logits)
assert(torch.allclose(out[:,2], out3[:,5]))
assert(torch.allclose(out[:,5], out3[:,2]))




conv_model = ConvModel([5], 5)
conv_op_model = OrderPreservingModel(conv_model, invariant=False)
out2 = conv_op_model(logits)
_,ind1 = torch.sort(logits)
_,ind2 = torch.sort(out2)
assert(torch.all(ind1 == ind2))
print("OK")

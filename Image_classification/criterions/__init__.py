from .cross_entropy_loss import CrossEntropyLoss
from .various_divergence import VariousDivergence
from .matry_CE import Matry_CrossEntropyLoss
from .nc1 import NC1
criterion_list = {
    "cross_entropy": CrossEntropyLoss,
    "various_divergence": VariousDivergence,
    "matry_CE": Matry_CrossEntropyLoss,
    "nc1": NC1
}

def build_criterion(args):
    if criterion_list.get(args.criterion, None) is not None:
        return criterion_list[args.criterion](args)
    else:
        raise NameError(f"Undefined criterion for {args.criterion}!")

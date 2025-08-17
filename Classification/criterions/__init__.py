from .cross_entropy_loss import CrossEntropyLoss
from .various_divergence import VariousDivergence
from .dual_space_kd_with_cross_model_attention import DualSpaceKDWithCMA
from .universal_logit_distillation import UniversalLogitDistillation
from .min_edit_dis_kld import MinEditDisForwardKLD
from .multi_level_ot import MULTI_LEVEL_OT
from .matry_CE import Matry_CrossEntropyLoss
from .nc1 import NC1
criterion_list = {
    "cross_entropy": CrossEntropyLoss,
    "various_divergence": VariousDivergence,
    "dual_space_kd_with_cross_model_attention": DualSpaceKDWithCMA,
    "universal_logit_distillation": UniversalLogitDistillation,
    "min_edit_dis_kld": MinEditDisForwardKLD,
    "min_edit_dis_kld": MinEditDisForwardKLD,
    "multi_level_ot": MULTI_LEVEL_OT,
    "matry_CE": Matry_CrossEntropyLoss,
    "nc1": NC1
}

def build_criterion(args):
    if criterion_list.get(args.criterion, None) is not None:
        return criterion_list[args.criterion](args)
    else:
        raise NameError(f"Undefined criterion for {args.criterion}!")

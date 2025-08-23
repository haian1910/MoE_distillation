from .sts_loss import STSLoss
from .dual_space_kd_with_cross_model_attention import DualSpaceKDWithCMA
from .rmse_cka import RMSE_CKA
from .ot_pro import OT_PRO
from .ot_pro_rmse_cka import OT_PRO_RMSE_CKA
from .min_cka import MIN_CKA
from .mmd import MMD
from .mmd_moe import  MMD_MOE
from .moe import MOE
from .cka_moe import CKA_MOE
from .moe_sts_loss import MoE_STSLoss
criterion_list = {
    "sts_loss": STSLoss,
    "dual_space_kd_with_cross_model_attention": DualSpaceKDWithCMA,
    "rmse_cka": RMSE_CKA,
    "ot_pro": OT_PRO,
    "ot_pro_rmse_cka": OT_PRO_RMSE_CKA,
    "min_cka": MIN_CKA,
    "mmd": MMD,
    "mmd_moe": MMD_MOE,
    "moe": MOE,
    "cka_moe": CKA_MOE,
    "moe_sts_loss": MoE_STSLoss
}

def build_criterion(args):
    if criterion_list.get(args.criterion, None) is not None:
        return criterion_list[args.criterion](args)
    else:
        raise NameError(f"Undefined criterion for {args.criterion}!")

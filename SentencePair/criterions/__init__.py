from .cross_entropy_loss import CrossEntropyLoss
from .various_divergence import VariousDivergence
from .dual_space_kd import DualSpaceKD
from .dual_space_kd_with_cross_model_attention import DualSpaceKDWithCMA
from .universal_logit_distillation import UniversalLogitDistillation
from .min_edit_dis_kld import MinEditDisForwardKLD
from .rmse_cka import RMSE_CKA
from .ot_pro import OT_PRO
from .ot_pro_rmse_cka import OT_PRO_RMSE_CKA
from .multi_level_ot import MULTI_LEVEL_OT
from .rank_cka import RANK_CKA
from .min_cka import MIN_CKA
from .cross_entropy_loss_moe import CrossEntropyLossMoE
from .moe import MOE
from .mmd import MMD
from .mmd_moe import MMD_MOE
from .mmd_moe_2tea import MMD_MOE_2TEA
from .topk_cka import TOPK_CKA
criterion_list = {
    "cross_entropy": CrossEntropyLoss,
    "various_divergence": VariousDivergence,
    "dual_space_kd": DualSpaceKD,
    "dual_space_kd_with_cross_model_attention": DualSpaceKDWithCMA,
    "universal_logit_distillation": UniversalLogitDistillation,
    "min_edit_dis_kld": MinEditDisForwardKLD,
    "rmse_cka": RMSE_CKA,
    "min_edit_dis_kld": MinEditDisForwardKLD,
    "ot_pro": OT_PRO,
    "ot_pro_rmse_cka": OT_PRO_RMSE_CKA,
    "multi_level_ot": MULTI_LEVEL_OT,
    "rank_cka": RANK_CKA,
    "min_cka": MIN_CKA,
    "moe": MOE,
    "ross_entropy_loss_moe": CrossEntropyLossMoE,
    "mmd": MMD,
    "mmd_moe": MMD_MOE,
    "mmd_moe_2tea": MMD_MOE_2TEA,
    "topk_cka": TOPK_CKA
}

def build_criterion(args):
    if criterion_list.get(args.criterion, None) is not None:
        return criterion_list[args.criterion](args)
    else:
        raise NameError(f"Undefined criterion for {args.criterion}!")

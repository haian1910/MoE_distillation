from .sts_loss import STSLoss
from .dual_space_kd_with_cross_model_attention import DualSpaceKDWithCMA
from .rmse_cka import RMSE_CKA
from .ot_pro import OT_PRO
from .ot_pro_rmse_cka import OT_PRO_RMSE_CKA
from .min_cka import MIN_CKA
from .multiple_negatives_ranking_loss import MultipleNegativesRankingLoss
from .universal_logit_distillation import UniversalLogitDistillation
from .multi_level_ot import MULTI_LEVEL_OT

criterion_list = {
    "sts_loss": STSLoss,
    "dual_space_kd_with_cross_model_attention": DualSpaceKDWithCMA,
    "rmse_cka": RMSE_CKA,
    "ot_pro": OT_PRO,
    "ot_pro_rmse_cka": OT_PRO_RMSE_CKA,
    "min_cka": MIN_CKA,
    "multiple_negatives_ranking_loss": MultipleNegativesRankingLoss,
    "universal_logit_distillation": UniversalLogitDistillation,
    "multi_level_ot": MULTI_LEVEL_OT
}

def build_criterion(args):
    if criterion_list.get(args.criterion, None) is not None:
        return criterion_list[args.criterion](args)
    else:
        raise NameError(f"Undefined criterion for {args.criterion}!")

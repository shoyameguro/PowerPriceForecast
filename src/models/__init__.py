from .lgbm_model import LGBMWrapper
from .xgb_model import XGBWrapper
from .nlinear_model import NLinearWrapper

MODEL_REGISTRY = {
    "lgbm": LGBMWrapper,
    "xgb": XGBWrapper,
    "nlinear": NLinearWrapper,
}


from .lgbm_model import LGBMWrapper
from .xgb_model import XGBWrapper

MODEL_REGISTRY = {
    "lgbm": LGBMWrapper,
    "xgb": XGBWrapper,
}


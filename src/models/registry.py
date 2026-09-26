"""
Common dispatch interface for the three architectures.

Stage 3 training and Stage 4 compression code should build models through
`build_model()` rather than importing cnn1d/gru/tcn directly, so adding a
fourth architecture later only requires registering it here.
"""

from src.models.cnn1d import build_cnn1d
from src.models.gru import build_gru
from src.models.tcn import build_tcn

MODEL_BUILDERS = {
    "cnn1d": build_cnn1d,
    "gru": build_gru,
    "tcn": build_tcn,
}


def build_model(name: str, input_shape: tuple[int, int], num_classes: int, config: dict):
    """Build a registered architecture by name, e.g. build_model("tcn", (256, 410), 34, cfg["tcn"])."""
    if name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown architecture '{name}'. Registered: {list(MODEL_BUILDERS)}")
    return MODEL_BUILDERS[name](input_shape, num_classes, config)

import os


def _get_env_value(name, default):
    raw_value = os.getenv(name)
    if raw_value is None or raw_value == "":
        return default
    if isinstance(default, bool):
        return raw_value.lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int) and not isinstance(default, bool):
        return int(raw_value)
    if isinstance(default, float):
        return float(raw_value)
    return raw_value


class Config:
    # ---------------------------------------------------------
    # Global Hyperparameters
    # ---------------------------------------------------------
    
    # Training Parameters
    LEARNING_RATE = _get_env_value("VV_LEARNING_RATE", 1e-3)
    BATCH_SIZE = _get_env_value("VV_BATCH_SIZE", 32)
    MAX_EPOCHS = _get_env_value("VV_MAX_EPOCHS", 100)
    WEIGHT_DECAY = _get_env_value("VV_WEIGHT_DECAY", 1e-4)
    
    # Dataset Parameters
    DATA_DIR = _get_env_value(
        "VV_DATA_DIR",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"),
    )
    NUM_WORKERS = _get_env_value("VV_NUM_WORKERS", 4)
    
    # Model Architecture Parameters
    INPUT_DIM = _get_env_value("VV_INPUT_DIM", 64)
    HIDDEN_DIM = _get_env_value("VV_HIDDEN_DIM", 128)
    OUTPUT_DIM = _get_env_value("VV_OUTPUT_DIM", 64)
    
    # Feature Extraction Parameters
    N_MELS = _get_env_value("VV_N_MELS", 64)
    SAMPLE_RATE = _get_env_value("VV_SAMPLE_RATE", 16000)
    N_EIGENMODES = _get_env_value("VV_N_EIGENMODES", 64)

# Create a global instance to be imported
cfg = Config()

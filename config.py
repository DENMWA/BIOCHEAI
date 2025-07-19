import os

class BaseConfig:
    APP_NAME = "BioCheAI"
    VERSION = "5.0.1"
    MODEL_DIR = "models/"
    DATA_DIR = "data/"
    LOGGING_LEVEL = "INFO"
    ENABLE_SELF_RETRAINING = True
    ALLOW_USER_UPLOADS = True
    DEFAULT_MODEL = {
        "DNA": "models/dna/default_model.pkl",
        "RNA": "models/rna/default_model.pkl",
        "PTM": "models/ptm/ptm_predictor.pkl"
    }

class DevConfig(BaseConfig):
    DEBUG = True
    LOGGING_LEVEL = "DEBUG"
    TEST_MODE = True

class ProdConfig(BaseConfig):
    DEBUG = False
    LOGGING_LEVEL = "WARNING"
    TEST_MODE = False

# Determine environment from ENV_MODE environment variable
env_mode = os.getenv("ENV_MODE", "dev").lower()

if env_mode == "prod":
    Config = ProdConfig()
else:
    Config = DevConfig()
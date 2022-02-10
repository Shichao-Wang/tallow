import logging

# for reproducibility
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from .common.importing import try_import_and_raise
from .common.logging import TallowLogger, TqdmLoggingHandler

logging.setLoggerClass(TallowLogger)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
    handlers=[TqdmLoggingHandler()],
)


try_import_and_raise("torch")

from .engine import NativeEngine
from .misc import seed_all

# isort: list
__all__ = ["Embedding", "NativeEngine", "SequenceEmbedding", "seed_all"]

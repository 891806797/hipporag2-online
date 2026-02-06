from .base import EmbeddingConfig, BaseEmbeddingModel
from .OpenAI import OpenAIEmbeddingModel

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def _get_embedding_model_class(embedding_model_name: str = "text-embedding-3-small"):
    """
    根据嵌入模型名称获取嵌入模型类。

    Args:
        embedding_model_name (str): 嵌入模型名称，默认为"text-embedding-3-small"。

    Returns:
        BaseEmbeddingModel: 嵌入模型类。

    Raises:
        AssertionError: 如果未知的嵌入模型名称。
    """
    if "text-embedding" in embedding_model_name:
        return OpenAIEmbeddingModel
    assert False, f"未知的嵌入模型名称: {embedding_model_name}"

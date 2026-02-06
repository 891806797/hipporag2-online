from copy import deepcopy
from typing import List, Optional

import numpy as np
from tqdm import tqdm
from openai import OpenAI


from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger
from .base import BaseEmbeddingModel, EmbeddingConfig, make_cache_embed

logger = get_logger(__name__)

class OpenAIEmbeddingModel(BaseEmbeddingModel):
    """OpenAI嵌入模型实现。"""

    def __init__(self, global_config: Optional[BaseConfig] = None, embedding_model_name: Optional[str] = None) -> None:
        """
        初始化OpenAI嵌入模型。

        Args:
            global_config (Optional[BaseConfig]): 全局配置对象。默认为None。
            embedding_model_name (Optional[str]): 嵌入模型名称。默认为None。
        """
        super().__init__(global_config=global_config)

        if embedding_model_name is not None:
            self.embedding_model_name = embedding_model_name
            logger.debug(
                f"使用{self.__class__.__name__}的embedding_model_name覆盖为: {self.embedding_model_name}")

        self._init_embedding_config()

        # 初始化嵌入模型
        logger.debug(
            f"使用以下参数初始化{self.__class__.__name__}的嵌入模型: {self.embedding_config.model_init_params}")

        self.embedding_api_key = self.global_config.embedding_api_key

        # 使用配置中的API密钥初始化OpenAI客户端
        self.client = OpenAI(
            base_url=self.global_config.embedding_base_url,
            api_key=self.embedding_api_key
        )


    def _init_embedding_config(self) -> None:
        """
        提取嵌入模型特定参数以初始化EmbeddingConfig。

        Returns:
            None
        """

        config_dict = {
            "embedding_model_name": self.embedding_model_name,
            "norm": self.global_config.embedding_return_as_normalized,
            # "max_seq_length": self.global_config.embedding_max_seq_len,
            "model_init_params": {
                # "model_name_or_path": self.embedding_model_name2mode_name_or_path[self.embedding_model_name],
                "pretrained_model_name_or_path": self.embedding_model_name,
                "trust_remote_code": True,
                # "torch_dtype": "auto",
                # 'device_map': "auto",  # 添加此行以使用多个GPU
                # **kwargs
            },
            "encode_params": {
                "max_length": self.global_config.embedding_max_seq_len,  # 官方示例中的32768，
                "instruction": "",
                "batch_size": self.global_config.embedding_batch_size,
                "num_workers": 32
            },
        }

        self.embedding_config = EmbeddingConfig.from_dict(config_dict=config_dict)
        logger.debug(f"初始化{self.__class__.__name__}的embedding_config: {self.embedding_config}")

    def encode(self, texts: List[str]):
        """
        编码文本列表。

        Args:
            texts (List[str]): 要编码的文本列表。

        Returns:
            numpy.ndarray: 嵌入向量数组。
        """
        texts = [t.replace("\n", " ") for t in texts]
        texts = [t if t != '' else ' ' for t in texts]
        response = self.client.embeddings.create(input=texts, model=self.embedding_model_name)
        results = np.array([v.embedding for v in response.data])

        return results

    def batch_encode(self, texts: List[str], **kwargs) -> None:
        """
        批量编码文本。

        Args:
            texts (List[str]): 要编码的文本列表。
            **kwargs: 其他关键字参数。

        Returns:
            numpy.ndarray: 嵌入向量数组。
        """
        if isinstance(texts, str): texts = [texts]

        params = deepcopy(self.embedding_config.encode_params)
        if kwargs: params.update(kwargs)

        if "instruction" in kwargs:
            if kwargs["instruction"] != '':
                params["instruction"] = f"指令: {kwargs['instruction']}\n查询: "
            # del params["instruction"]

        logger.debug(f"使用{self.__class__.__name__}调用，参数:\n{params}")

        batch_size = params.pop("batch_size", 16)

        if len(texts) <= batch_size:
            results = self.encode(texts)
        else:
            pbar = tqdm(total=len(texts), desc="批量编码")
            results = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                try:
                    results.append(self.encode(batch))
                except:
                    import ipdb; ipdb.set_trace()
                pbar.update(batch_size)
            pbar.close()
            results = np.concatenate(results)

        if self.embedding_config.norm:
            results = (results.T / np.linalg.norm(results, axis=1)).T

        return results

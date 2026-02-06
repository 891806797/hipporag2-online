import functools
import hashlib
import json
import os
import sqlite3
from copy import deepcopy
from typing import List, Tuple

import httpx
import openai
from filelock import FileLock
from openai import OpenAI

from packaging import version
from tenacity import retry, stop_after_attempt, wait_fixed

from ..utils.config_utils import BaseConfig
from ..utils.llm_utils import (
    TextChatMessage
)
from ..utils.logging_utils import get_logger
from .base import BaseLLM, LLMConfig

logger = get_logger(__name__)

def cache_response(func):
    """
    缓存响应的装饰器。

    Args:
        func: 要装饰的函数。

    Returns:
        包装后的函数，使用SQLite缓存响应。
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # 从args或kwargs获取messages
        if args:
            messages = args[0]
        else:
            messages = kwargs.get("messages")
        if messages is None:
            raise ValueError("缓存缺少必需的'messages'参数。")

        # 从kwargs或self.llm_config.generate_params获取model、seed和temperature
        gen_params = getattr(self, "llm_config", {}).generate_params if hasattr(self, "llm_config") else {}
        model = kwargs.get("model", gen_params.get("model"))
        seed = kwargs.get("seed", gen_params.get("seed"))
        temperature = kwargs.get("temperature", gen_params.get("temperature"))

        # 构建key数据，转换为JSON字符串并哈希生成key_hash
        key_data = {
            "messages": messages,  # messages需要JSON可序列化
            "model": model,
            "seed": seed,
            "temperature": temperature,
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()

        # 锁的文件名，确保并发访问时的互斥
        lock_file = self.cache_file_name + ".lock"

        # 尝试从SQLite缓存读取
        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            # 如果表不存在，则创建它
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """)
            conn.commit()  # 提交以保存表创建
            c.execute("SELECT message, metadata FROM cache WHERE key = ?", (key_hash,))
            row = c.fetchone()
            conn.close()
            if row is not None:
                message, metadata_str = row
                metadata = json.loads(metadata_str)
                # 返回缓存结果并标记为命中
                return message, metadata, True

        # 如果缓存未命中，调用原始函数获取结果
        result = func(self, *args, **kwargs)
        message, metadata = result

        # 将新结果插入缓存
        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            # 再次确保表存在（如果不存在则创建）
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """)
            metadata_str = json.dumps(metadata)
            c.execute("INSERT OR REPLACE INTO cache (key, message, metadata) VALUES (?, ?, ?)",
                      (key_hash, message, metadata_str))
            conn.commit()
            conn.close()

        return message, metadata, False

    return wrapper

def dynamic_retry_decorator(func):
    """
    动态重试装饰器。

    Args:
        func: 要装饰的函数。

    Returns:
        包装后的函数，使用tenacity进行重试。
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        max_retries = getattr(self, "max_retries", 5)
        dynamic_retry = retry(stop=stop_after_attempt(max_retries), wait=wait_fixed(1))
        decorated_func = dynamic_retry(func)
        return decorated_func(self, *args, **kwargs)
    return wrapper

class CacheOpenAI(BaseLLM):
    """OpenAI LLM实现。"""
    @classmethod
    def from_experiment_config(cls, global_config: BaseConfig) -> "CacheOpenAI":
        """
        从实验配置创建CacheOpenAI实例。

        Args:
            global_config (BaseConfig): 全局配置对象。

        Returns:
            CacheOpenAI: CacheOpenAI实例。
        """
        config_dict = global_config.__dict__
        config_dict['max_retries'] = global_config.max_retry_attempts
        cache_dir = os.path.join(global_config.save_dir, "llm_cache")
        return cls(cache_dir=cache_dir, global_config=global_config)

    def __init__(self, cache_dir, global_config, cache_filename: str = None,
                 high_throughput: bool = True,
                 **kwargs) -> None:
        """
        初始化CacheOpenAI实例。

        Args:
            cache_dir (str): 缓存目录路径。
            global_config (BaseConfig): 全局配置对象。
            cache_filename (str, optional): 缓存文件名。默认为None。
            high_throughput (bool, optional): 是否启用高吞吐量。默认为True。
            **kwargs: 其他关键字参数。
        """
        super().__init__()
        self.cache_dir = cache_dir
        self.global_config = global_config

        self.llm_name = global_config.llm_name
        self.llm_base_url = global_config.llm_base_url
        self.llm_api_key = global_config.llm_api_key

        os.makedirs(self.cache_dir, exist_ok=True)
        if cache_filename is None:
            cache_filename = f"{self.llm_name.replace('/', '_')}_cache.sqlite"
        self.cache_file_name = os.path.join(self.cache_dir, cache_filename)

        self._init_llm_config()
        if high_throughput:
            limits = httpx.Limits(max_connections=500, max_keepalive_connections=100)
            client = httpx.Client(limits=limits, timeout=httpx.Timeout(5*60, read=5*60))
        else:
            client = None

        self.max_retries = kwargs.get("max_retries", 2)

        # 使用配置中的API密钥初始化OpenAI客户端
        self.openai_client = OpenAI(
            base_url=self.llm_base_url,
            api_key=self.llm_api_key,
            http_client=client,
            max_retries=self.max_retries
        )

    def _init_llm_config(self) -> None:
        """
        初始化LLM配置。

        Returns:
            None
        """
        config_dict = self.global_config.__dict__

        config_dict['generate_params'] = {
                "model": self.global_config.llm_name,
                "max_completion_tokens": config_dict.get("max_new_tokens", 400),
                "n": config_dict.get("num_gen_choices", 1),
                "seed": config_dict.get("seed", 0),
                "temperature": config_dict.get("temperature", 0.0),
            }

        self.llm_config = LLMConfig.from_dict(config_dict=config_dict)
        logger.debug(f"初始化{self.__class__.__name__}的llm_config: {self.llm_config}")

    @cache_response
    @dynamic_retry_decorator
    def infer(
        self,
        messages: List[TextChatMessage],
        **kwargs
    ) -> Tuple[List[TextChatMessage], dict]:
        """
        执行LLM推理。

        Args:
            messages (List[TextChatMessage]): 聊天消息列表。
            **kwargs: 其他关键字参数。

        Returns:
            Tuple[List[TextChatMessage], dict]: 包含响应消息和元数据的元组。
        """
        params = deepcopy(self.llm_config.generate_params)
        if kwargs:
            params.update(kwargs)
        params["messages"] = messages
        logger.debug(f"使用以下参数调用OpenAI GPT API:\n{params}")

        if 'gpt' not in params['model'] or version.parse(openai.__version__) < version.parse("1.45.0"): # 如果我们使用vllm调用openai api或者如果我们使用openai但版本太旧而无法使用'max_completion_tokens'参数
            # TODO openai协议中的奇怪版本变化，但我们当前的vllm版本尚未更改
            params['max_tokens'] = params.pop('max_completion_tokens')

        response = self.openai_client.chat.completions.create(**params)

        response_message = response.choices[0].message.content
        assert isinstance(response_message, str), "response_message应该是一个字符串"

        metadata = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "finish_reason": response.choices[0].finish_reason,
        }

        return response_message, metadata

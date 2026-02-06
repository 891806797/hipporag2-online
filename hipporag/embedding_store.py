import numpy as np
from tqdm import tqdm
import os
from typing import Union, Optional, List, Dict, Set, Any, Tuple, Literal
import logging
from copy import deepcopy
import pandas as pd

from .utils.misc_utils import compute_mdhash_id, NerRawOutput, TripleRawOutput

logger = logging.getLogger(__name__)

class EmbeddingStore:
    """
    嵌入存储类，用于管理文本嵌入的存储和检索。
    """
    def __init__(self, embedding_model, db_filename, batch_size, namespace):
        """
        使用必要的配置初始化类并设置工作目录。

        Parameters:
        embedding_model: 用于嵌入的模型。
        db_filename: 数据存储或检索的目录路径。
        batch_size: 用于处理的批处理大小。
        namespace: 用于数据隔离的唯一标识符。

        Functionality:
        - 将提供的参数赋值给实例变量。
        - 检查`db_filename`指定的目录是否存在。
          - 如果不存在，创建目录并记录操作。
        - 构建用于以parquet文件格式存储数据的文件名。
        - 调用方法`_load_data()`以初始化数据加载过程。
        """
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.namespace = namespace

        if not os.path.exists(db_filename):
            logger.info(f"创建工作目录: {db_filename}")
            os.makedirs(db_filename, exist_ok=True)

        self.filename = os.path.join(
            db_filename, f"vdb_{self.namespace}.parquet"
        )
        self._load_data()

    def get_missing_string_hash_ids(self, texts: List[str]):
        """
        获取缺失的字符串哈希ID。

        Args:
            texts (List[str]): 要检查的文本列表。

        Returns:
            Dict: 缺失的哈希ID到其内容的映射。
        """
        nodes_dict = {}

        for text in texts:
            nodes_dict[compute_mdhash_id(text, prefix=self.namespace + "-")] = {'content': text}

        # 从输入字典中获取所有hash_ids
        all_hash_ids = list(nodes_dict.keys())
        if not all_hash_ids:
            return  {}

        existing = self.hash_id_to_row.keys()

        # 过滤出缺失的hash_ids
        missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing]
        texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]

        return {h: {"hash_id": h, "content": t} for h, t in zip(missing_ids, texts_to_encode)}

    def insert_strings(self, texts: List[str]):
        """
        插入字符串到存储中。

        Args:
            texts (List[str]): 要插入的文本列表。
        """
        nodes_dict = {}

        for text in texts:
            nodes_dict[compute_mdhash_id(text, prefix=self.namespace + "-")] = {'content': text}

        # 从输入字典中获取所有hash_ids
        all_hash_ids = list(nodes_dict.keys())
        if not all_hash_ids:
            return  # 没有要插入的内容

        existing = self.hash_id_to_row.keys()

        # 过滤出缺失的hash_ids
        missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing]

        logger.info(
            f"插入{len(missing_ids)}条新记录，{len(all_hash_ids) - len(missing_ids)}条记录已存在。")

        if not missing_ids:
            return  {}# 所有记录已存在

        # 准备要从"content"字段编码的文本
        texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]

        missing_embeddings = self.embedding_model.batch_encode(texts_to_encode)

        self._upsert(missing_ids, texts_to_encode, missing_embeddings)

    def _load_data(self):
        """
        从文件加载数据。
        """
        if os.path.exists(self.filename):
            df = pd.read_parquet(self.filename)
            self.hash_ids, self.texts, self.embeddings = df["hash_id"].values.tolist(), df["content"].values.tolist(), df["embedding"].values.tolist()
            self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
            self.hash_id_to_row = {
                h: {"hash_id": h, "content": t}
                for h, t in zip(self.hash_ids, self.texts)
            }
            self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
            self.text_to_hash_id = {self.texts[idx]: h  for idx, h in enumerate(self.hash_ids)}
            assert len(self.hash_ids) == len(self.texts) == len(self.embeddings)
            logger.info(f"从{self.filename}加载{len(self.hash_ids)}条记录")
        else:
            self.hash_ids, self.texts, self.embeddings = [], [], []
            self.hash_id_to_idx, self.hash_id_to_row = {}, {}

    def _save_data(self):
        """
        保存数据到文件。
        """
        data_to_save = pd.DataFrame({
            "hash_id": self.hash_ids,
            "content": self.texts,
            "embedding": self.embeddings
        })
        data_to_save.to_parquet(self.filename, index=False)
        self.hash_id_to_row = {h: {"hash_id": h, "content": t} for h, t, e in zip(self.hash_ids, self.texts, self.embeddings)}
        self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
        self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
        self.text_to_hash_id = {self.texts[idx]: h for idx, h in enumerate(self.hash_ids)}
        logger.info(f"将{len(self.hash_ids)}条记录保存到{self.filename}")

    def _upsert(self, hash_ids, texts, embeddings):
        """
        更新或插入数据。

        Args:
            hash_ids: 哈希ID列表。
            texts: 文本列表。
            embeddings: 嵌入列表。
        """
        self.embeddings.extend(embeddings)
        self.hash_ids.extend(hash_ids)
        self.texts.extend(texts)

        logger.info(f"保存新记录。")
        self._save_data()

    def delete(self, hash_ids):
        """
        删除指定的哈希ID。

        Args:
            hash_ids: 要删除的哈希ID列表。
        """
        indices = []

        for hash in hash_ids:
            indices.append(self.hash_id_to_idx[hash])

        sorted_indices = np.sort(indices)[::-1]

        for idx in sorted_indices:
            self.hash_ids.pop(idx)
            self.texts.pop(idx)
            self.embeddings.pop(idx)

        logger.info(f"删除后保存记录。")
        self._save_data()

    def get_row(self, hash_id):
        """
        获取指定哈希ID的行。

        Args:
            hash_id: 哈希ID。

        Returns:
            Dict: 行数据。
        """
        return self.hash_id_to_row[hash_id]

    def get_hash_id(self, text):
        """
        获取指定文本的哈希ID。

        Args:
            text: 文本内容。

        Returns:
            str: 哈希ID。
        """
        return self.text_to_hash_id[text]

    def get_rows(self, hash_ids, dtype=np.float32):
        """
        获取指定哈希ID的行。

        Args:
            hash_ids: 哈希ID列表。
            dtype: 数据类型。默认为np.float32。

        Returns:
            Dict: 哈希ID到行数据的映射。
        """
        if not hash_ids:
            return {}

        results = {id : self.hash_id_to_row[id] for id in hash_ids}

        return results

    def get_all_ids(self):
        """
        获取所有哈希ID。

        Returns:
            List: 所有哈希ID的深拷贝列表。
        """
        return deepcopy(self.hash_ids)

    def get_all_id_to_rows(self):
        """
        获取所有ID到行的映射。

        Returns:
            Dict: 所有ID到行数据的深拷贝映射。
        """
        return deepcopy(self.hash_id_to_row)

    def get_all_texts(self):
        """
        获取所有文本。

        Returns:
            Set: 所有文本的集合。
        """
        return set(row['content'] for row in self.hash_id_to_row.values())

    def get_embedding(self, hash_id, dtype=np.float32) -> np.ndarray:
        """
        获取指定哈希ID的嵌入。

        Args:
            hash_id: 哈希ID。
            dtype: 数据类型。默认为np.float32。

        Returns:
            np.ndarray: 嵌入向量。
        """
        return self.embeddings[self.hash_id_to_idx[hash_id]].astype(dtype)

    def get_embeddings(self, hash_ids, dtype=np.float32) -> list[np.ndarray]:
        """
        获取指定哈希ID的嵌入列表。

        Args:
            hash_ids: 哈希ID列表。
            dtype: 数据类型。默认为np.float32。

        Returns:
            list[np.ndarray]: 嵌入向量列表。
        """
        if not hash_ids:
            return []

        indices = np.array([self.hash_id_to_idx[h] for h in hash_ids], dtype=np.intp)
        embeddings = np.array(self.embeddings, dtype=dtype)[indices]

        return embeddings

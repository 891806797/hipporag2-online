import os
from dataclasses import dataclass, field
from typing import (
    Literal,
    Union,
    Optional
)

from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class BaseConfig:
    """唯一配置类。"""
    # LLM特定属性
    llm_name: str = field(
        default="gpt-4o-mini",
        metadata={"help": "指示使用哪个LLM模型的类名。"}
    )
    llm_api_key: str = field(
        default=None,
        metadata={"help": "用于LLM API调用的密钥。"}
    )
    llm_base_url: str = field(
        default=None,
        metadata={"help": "LLM模型的基础URL，如果为空，则使用OpenAI服务。"}
    )
    embedding_api_key: str = field(
        default=None,
        metadata={"help": "用于向量化API调用的密钥。"}
    )
    embedding_base_url: str = field(
        default=None,
        metadata={"help": "OpenAI兼容向量化模型的基础URL，如果为空，则使用OpenAI服务。"}
    )
    azure_endpoint: str = field(
        default=None,
        metadata={"help": "LLM模型的Azure端点URI，如果为空，则直接使用OpenAI服务。"}
    )
    azure_embedding_endpoint: str = field(
        default=None,
        metadata={"help": "OpenAI向量化模型的Azure端点URI，如果为空，则直接使用OpenAI服务。"}
    )
    max_new_tokens: Union[None, int] = field(
        default=2048,
        metadata={"help": "每次推理生成的最大新token数。"}
    )
    num_gen_choices: int = field(
        default=1,
        metadata={"help": "为每个输入消息生成多少个聊天完成选项。"}
    )
    seed: Union[None, int] = field(
        default=None,
        metadata={"help": "随机种子。"}
    )
    temperature: float = field(
        default=0,
        metadata={"help": "每次推理采样的温度。"}
    )
    response_format: Union[dict, None] = field(
        default_factory=lambda: { "type": "json_object" },
        metadata={"help": "指定模型必须输出的格式。"}
    )

    ## LLM特定属性 -> 异步超参数
    max_retry_attempts: int = field(
        default=5,
        metadata={"help": "异步API调用的最大重试次数。"}
    )
    # 存储特定属性
    force_openie_from_scratch: bool = field(
        default=False,
        metadata={"help": "如果设置为True，将忽略所有现有的openie文件并从头重建。"}
    )

    # 存储特定属性
    force_index_from_scratch: bool = field(
        default=False,
        metadata={"help": "如果设置为True，将忽略所有现有的存储文件和图数据并从头重建。"}
    )
    rerank_dspy_file_path: str = field(
        default=None,
        metadata={"help": "重排序dspy文件的路径。"}
    )
    passage_node_weight: float = field(
        default=0.05,
        metadata={"help": "修改PPR中段落节点权重的乘法因子。"}
    )
    save_openie: bool = field(
        default=True,
        metadata={"help": "如果设置为True，将把OpenIE模型保存到磁盘。"}
    )

    # 预处理特定属性
    text_preprocessor_class_name: str = field(
        default="TextPreprocessor",
        metadata={"help": "在预处理中使用的基于文本的预处理器的名称。"}
    )
    preprocess_encoder_name: str = field(
        default="gpt-4o",
        metadata={"help": "在预处理中使用的编码器名称（目前专门为文档分块实现）。"}
    )
    preprocess_chunk_overlap_token_size: int = field(
        default=128,
        metadata={"help": "相邻分块之间的重叠token数。"}
    )
    preprocess_chunk_max_token_size: int = field(
        default=None,
        metadata={"help": "每个分块可以包含的最大token数。如果设置为None，整个文档将被视为单个分块。"}
    )
    preprocess_chunk_func: Literal["by_token", "by_word"] = field(default='by_token')


    # 信息提取特定属性
    information_extraction_model_name: Literal["openie_openai_gpt", ] = field(
        default="openie_openai_gpt",
        metadata={"help": "指示使用哪个信息提取模型的类名。"}
    )
    openie_mode: Literal["offline", "online"] = field(
        default="online",
        metadata={"help": "要使用的OpenIE模型模式。"}
    )
    skip_graph: bool = field(
        default=False,
        metadata={"help": "是否跳过图构建。在首次运行vllm离线索引时设置为true。"}
    )


    # 向量化特定属性
    embedding_model_name: str = field(
        default="text-embedding-3-small",
        metadata={"help": "指示使用哪个向量化模型的类名。"}
    )
    embedding_batch_size: int = field(
        default=16,
        metadata={"help": "调用向量化模型的批处理大小。"}
    )
    embedding_return_as_normalized: bool = field(
        default=True,
        metadata={"help": "是否对编码的嵌入进行归一化。"}
    )
    embedding_max_seq_len: int = field(
        default=2048,
        metadata={"help": "向量化模型的最大序列长度。"}
    )
    embedding_model_dtype: Literal["float16", "float32", "bfloat16", "auto"] = field(
        default="auto",
        metadata={"help": "本地向量化模型的数据类型。"}
    )



    # 图构建特定属性
    synonymy_edge_topk: int = field(
        default=2047,
        metadata={"help": "构建同义词边时knn检索的k值。"}
    )
    synonymy_edge_query_batch_size: int = field(
        default=1000,
        metadata={"help": "构建同义词边时knn检索的查询嵌入批处理大小。"}
    )
    synonymy_edge_key_batch_size: int = field(
        default=10000,
        metadata={"help": "构建同义词边时knn检索的键嵌入批处理大小。"}
    )
    synonymy_edge_sim_threshold: float = field(
        default=0.8,
        metadata={"help": "包含候选同义词节点的相似度阈值。"}
    )
    is_directed_graph: bool = field(
        default=False,
        metadata={"help": "图是否为有向图。"}
    )



    # 检索特定属性
    linking_top_k: int = field(
        default=5,
        metadata={"help": "每次检索步骤链接的节点数"}
    )
    retrieval_top_k: int = field(
        default=200,
        metadata={"help": "每次检索步骤检索k个文档"}
    )
    damping: float = field(
        default=0.5,
        metadata={"help": "PPR算法的阻尼因子。"}
    )


    # QA特定属性
    max_qa_steps: int = field(
        default=1,
        metadata={"help": "回答单个问题时，我们使用检索和推理交替的最大步骤数。"}
    )
    qa_top_k: int = field(
        default=5,
        metadata={"help": "将前k个文档提供给QA模型进行阅读。"}
    )

    # 保存目录（最高级别目录）
    save_dir: str = field(
        default=None,
        metadata={"help": "保存所有相关信息的目录。如果提供，将覆盖所有默认save_dir设置。如果未提供，则如果不运行特定数据集，默认为`outputs`，否则默认为数据集自定义的输出目录。"}
    )



    # 数据集运行特定属性
    ## 数据集运行特定属性 -> 通用
    dataset: Optional[Literal['hotpotqa', 'hotpotqa_train', 'musique', '2wikimultihopqa']] = field(
        default=None,
        metadata={"help": "要使用的数据集。如果指定，意味着我们将运行特定数据集。如果未指定，意味着我们自由运行。"}
    )
    ## 数据集运行特定属性 -> 图
    graph_type: Literal[
        'dpr_only',
        'entity',
        'passage_entity', 'relation_aware_passage_entity',
        'passage_entity_relation',
        'facts_and_sim_passage_node_unidirectional',
    ] = field(
        default="facts_and_sim_passage_node_unidirectional",
        metadata={"help": "实验中使用的图类型。"}
    )
    corpus_len: Optional[int] = field(
        default=None,
        metadata={"help": "要使用的语料库长度。"}
    )


    def __post_init__(self):
        if self.save_dir is None: # 如果未提供save_dir
            if self.dataset is None: self.save_dir = 'outputs' # 自由运行
            else: self.save_dir = os.path.join('outputs', self.dataset) # 在此处自定义数据集的输出目录
        logger.debug(f"初始化最高级别的save_dir为{self.save_dir}")

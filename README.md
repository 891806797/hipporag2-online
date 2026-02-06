# HippoRAG 2: 从RAG到记忆

<p align="center">
    <img src="https://github.com/OSU-NLP-Group/HippoRAG/raw/main/images/hippo_brain.png" width="55%" style="max-width: 300px;">
</p>

[![arXiv](https://img.shields.io/badge/arXiv-2502.14802%20HippoRAG%202-b31b1b)](https://arxiv.org/abs/2502.14802)
[![arXiv](https://img.shields.io/badge/arXiv-2405.14831%20HippoRAG%201-b31b1b)](https://arxiv.org/abs/2405.14831)
[![GitHub](https://img.shields.io/badge/GitHub-HippoRAG%201-blue)](https://github.com/OSU-NLP-Group/HippoRAG/tree/legacy)

## 简介

HippoRAG 2是一个为大型语言模型（LLM）设计的强大记忆框架，它增强了LLM识别和利用新知识中连接的能力——这反映了人类长期记忆的一个关键功能。

我们的实验表明，HippoRAG 2在提高关联性（多跳检索）和意义构建（整合大型和复杂上下文的过程）方面，即使在最先进的RAG系统中，也不会牺牲其在简单任务上的性能。

与其前身一样，HippoRAG 2在在线过程中保持成本和延迟效率，同时与GraphRAG、RAPTOR和LightRAG等其他基于图的解决方案相比，离线索引使用的资源显著减少。

<p align="center">
  <img align="center" src="https://github.com/OSU-NLP-Group/HippoRAG/raw/main/images/intro.png" />
</p>
<p align="center">
  <b>图1：</b>在三个关键维度上评估持续学习能力：事实记忆（NaturalQuestions、PopQA）、意义构建（NarrativeQA）和关联性（MuSiQue、2Wiki、HotpotQA和LV-Eval）。HippoRAG 2在所有类别中都超越了其他方法，使其更接近真正的长期记忆。
</p>

<p align="center">
  <img align="center" src="https://github.com/OSU-NLP-Group/HippoRAG/raw/main/images/methodology.png" />
</p>
<p align="center">
  <b>图2：</b>HippoRAG 2方法。
</p>

## 本项目特点

本项目是对原HippoRAG 2的简化版本，主要特点包括：

- **仅支持在线模式**：简化了原项目，只保留了在线推理模式
- **OpenAI模型支持**：移除了OpenAI以外的LLM相关代码，专注于OpenAI API的使用
- **中文提示词**：将所有提示词翻译为中文，方便中文用户使用
- **核心算法保持一致**：与原HippoRAG 2保持相同的检索和问答算法

## 安装

```sh
conda create -n hipporag python=3.10
conda activate hipporag
pip install -r requirements.txt
```

初始化环境变量并激活环境：

```sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HOME=<Huggingface home目录路径>

conda activate hipporag
```

**注意**：API密钥不再需要通过环境变量配置，而是直接在BaseConfig中设置。

## 快速开始

### 基本使用示例

以下示例展示了如何使用HippoRAG进行文档索引、检索和问答：

```python
from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig

# 配置参数
config = BaseConfig(
    llm_name="qwen3-max",  # LLM模型名称
    llm_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # LLM API地址
    llm_api_key="your-api-key",  # LLM API密钥
    max_new_tokens=4096,  # 最大生成token数
    temperature=0.0,  # 温度参数
    embedding_model_name="text-embedding-v4",  # 嵌入模型名称
    embedding_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 嵌入API地址
    embedding_api_key="your-api-key",  # 嵌入API密钥
    graph_type="facts_and_sim_passage_node_unidirectional",  # 图类型
    save_dir="outputs",  # 保存目录
    save_openie=True  # 是否保存OpenIE结果
)

# 定义保存目录
save_dir = "outputs"

# 初始化HippoRAG实例
hipporag = HippoRAG(save_dir=save_dir, global_config=config)

# 准备文档
docs = [
    "王林从小聪明，是村子里公认的神童。",
    "铁柱是王林的小名，他身体瘦弱。",
    "恒岳派收下了王林作为弟子，从此开始修行之路。",
]

# 索引文档
hipporag.index(docs=docs)

# 准备查询
queries = [
    "谁是王林？他的主要人际关系是什么样？",
    "王林和铁柱的关系是什么？",
]

# 执行问答
query_solutions, responses, metadata = hipporag.rag_qa(queries=queries)

# 打印结果
print("=== 问答结果 ===")
for i in range(len(queries)):
    print(f"问题: {queries[i]}")
    print(f"答案: {query_solutions[i].answer}")
    print(f"响应: {responses[i]}")
    print(f"元数据: {metadata[i]}")
    print("=" * 50)
```

### 使用OpenAI模型

如果你想使用OpenAI官方模型，可以这样配置：

```python
from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig

# 配置OpenAI参数
config = BaseConfig(
    llm_name="gpt-4o-mini",  # OpenAI模型名称
    llm_base_url="https://api.openai.com/v1",  # OpenAI API地址
    llm_api_key="your-openai-api-key",  # OpenAI API密钥
    embedding_model_name="text-embedding-3-small",  # OpenAI嵌入模型
    embedding_base_url="https://api.openai.com/v1",  # OpenAI嵌入API地址
    embedding_api_key="your-openai-api-key",  # OpenAI API密钥
    save_dir="outputs",
    save_openie=True
)

# 初始化HippoRAG
hipporag = HippoRAG(save_dir="outputs", global_config=config)

# 准备文档
docs = [
    "Oliver Badman是一名政治家。",
    "George Rankin是一名政治家。",
    "Cinderella参加了皇家舞会。",
    "王子用丢失的水晶鞋搜寻王国。",
]

# 索引和问答
hipporag.index(docs=docs)
query_solutions, responses, metadata = hipporag.rag_qa(
    queries=["George Rankin的职业是什么？"]
)
print(f"答案: {query_solutions[0].answer}")
```

### 文档删除

HippoRAG支持删除已索引的文档：

```python
# 删除指定文档
hipporag.delete(docs_to_delete=["王林从小聪明，是村子里公认的神童。"])

# 重新执行问答，结果会基于删除后的文档
query_solutions, responses, metadata = hipporag.rag_qa(queries=queries)
```

### 带评估的问答

如果需要评估检索和问答性能，可以提供标准答案和标准文档：

```python
# 准备标准答案和标准文档
gold_answers = [
    ["王林是村子里公认的神童"],
    ["铁柱是王林的小名"]
]

gold_docs = [
    ["王林从小聪明，是村子里公认的神童。"],
    ["铁柱是王林的小名，他身体瘦弱。"]
]

# 执行带评估的问答
query_solutions, responses, metadata, retrieval_result, qa_result = hipporag.rag_qa(
    queries=queries,
    gold_docs=gold_docs,
    gold_answers=gold_answers
)

# 打印评估结果
print(f"检索评估结果: {retrieval_result}")
print(f"问答评估结果: {qa_result}")
```

## 测试

在为HippoRAG做出贡献时，请运行以下脚本以确保你的更改不会导致核心模块的意外行为。

这些脚本测试索引、图加载、文档删除和HippoRAG对象的增量更新。

### 运行测试

要测试HippoRAG，请先在测试脚本中配置好API密钥，然后运行：

```sh
conda activate hipporag

python hipporag_test.py
```

**注意**：API密钥需要在测试脚本中的BaseConfig里配置，而不是通过环境变量设置。

## 项目结构

```
📦 hipporag2-online
│-- 📂 hipporag
│   ├── 📂 embedding_model          # 所有嵌入模型的实现
│   │   ├── __init__.py
│   │   ├── base.py                 # 基础嵌入模型类
│   │   └── OpenAI.py              # OpenAI嵌入模型实现
│   ├── 📂 evaluation               # 所有评估指标的实现
│   │   ├── __init__.py
│   │   ├── base.py                 # 基础评估指标类
│   │   ├── qa_eval.py              # QA评估指标
│   │   ├── retrieval_eval.py       # 检索评估指标
│   │   └── README.md              # 评估模块文档
│   ├── 📂 information_extraction  # 所有信息提取模型的实现
│   │   ├── __init__.py
│   │   └── openie_openai.py       # 使用OpenAI GPT的OpenIE模型
│   ├── 📂 llm                      # 大语言模型推理的类
│   │   ├── __init__.py
│   │   ├── base.py                 # LLM推理的配置类和基础类
│   │   └── openai_gpt.py          # 使用OpenAI GPT推理的类
│   ├── 📂 prompts                  # 提示词模板和提示词模板管理器类
│   │   ├── __init__.py
│   │   ├── linking.py              # 链接指令
│   │   ├── prompt_template_manager.py  # 提示词模板管理器实现
│   │   ├── dspy_prompts/          # 过滤提示词
│   │   └── templates/             # 提示词模板管理器加载的所有提示词模板
│   │       ├── __init__.py
│   │       ├── triple_extraction.py
│   │       ├── ner.py
│   │       ├── ner_query.py
│   │       ├── rag_qa_musique.py
│   │       ├── ircot_hotpotqa.py
│   │       ├── ircot_musique.py
│   │       └── README.md
│   ├── 📂 utils                    # 本仓库中使用的所有实用函数
│   │   ├── __init__.py
│   │   ├── config_utils.py         # 所有模块使用的统一配置
│   │   ├── embed_utils.py         # 嵌入相关实用函数
│   │   ├── eval_utils.py          # 评估相关实用函数
│   │   ├── llm_utils.py           # LLM相关实用函数
│   │   ├── logging_utils.py       # 日志相关实用函数
│   │   ├── misc_utils.py          # 其他实用函数
│   │   ├── qa_utils.py           # QA相关实用函数
│   │   └── typing.py             # 类型定义
│   ├── __init__.py
│   ├── HippoRAG.py              # 用于启动检索、问答和评估的最高层类
│   ├── embedding_store.py       # 用于加载、管理和保存段落、实体和事实嵌入的存储数据库
│   └── rerank.py                # 重排序和过滤方法
│-- 📜 README.md                 # 本文档
│-- 📜 requirements.txt          # 依赖列表
│-- 📜 hipporag_test.py         # 测试脚本
```

## 核心功能

### 1. 文档索引

使用 [`HippoRAG.index()`](hipporag/HippoRAG.py:210) 方法对文档进行索引，构建知识图谱：

```python
hipporag.index(docs=docs)
```

索引过程包括：
- 开放信息提取（OpenIE）
- 实体和事实嵌入编码
- 知识图谱构建

### 2. 文档检索

使用 [`HippoRAG.retrieve()`](hipporag/HippoRAG.py:351) 方法检索相关文档：

```python
retrieval_results = hipporag.retrieve(queries=queries, num_to_retrieve=10)
```

检索过程包括：
- 事实检索
- 识别记忆
- 密集段落评分
- 基于个性化PageRank的重排序

### 3. 问答

使用 [`HippoRAG.rag_qa()`](hipporag/HippoRAG.py:439) 方法执行检索增强问答：

```python
query_solutions, responses, metadata = hipporag.rag_qa(queries=queries)
```

### 4. 评估

提供标准答案和标准文档时，系统会自动进行评估：

```python
query_solutions, responses, metadata, retrieval_result, qa_result = hipporag.rag_qa(
    queries=queries,
    gold_docs=gold_docs,
    gold_answers=gold_answers
)
```

评估指标包括：
- **检索评估**：Recall@k（召回率）
- **问答评估**：Exact Match（精确匹配）和F1 Score

## 配置参数

可以通过 [`BaseConfig`](hipporag/utils/config_utils.py:1) 类配置各种参数：

```python
from hipporag.utils.config_utils import BaseConfig

config = BaseConfig(
    llm_name="gpt-4o-mini",  # LLM模型名称
    llm_base_url="https://api.openai.com/v1",  # LLM API地址
    llm_api_key="your-api-key",  # LLM API密钥
    max_new_tokens=4096,  # 最大生成token数
    temperature=0.0,  # 温度参数
    embedding_model_name="text-embedding-3-small",  # 嵌入模型名称
    embedding_base_url="https://api.openai.com/v1",  # 嵌入API地址
    embedding_api_key="your-api-key",  # 嵌入API密钥
    graph_type="facts_and_sim_passage_node_unidirectional",  # 图类型
    save_dir="outputs",  # 保存目录
    save_openie=True  # 是否保存OpenIE结果
)
```

主要配置参数：

**LLM相关：**
- `llm_name`: LLM模型名称（如：gpt-4o-mini、qwen3-max等）
- `llm_base_url`: LLM API基础URL
- `llm_api_key`: LLM API密钥
- `max_new_tokens`: 最大生成token数，默认4096
- `temperature`: 温度参数，默认0.0

**嵌入模型相关：**
- `embedding_model_name`: 嵌入模型名称（如：text-embedding-3-small、nvidia/NV-Embed-v2等）
- `embedding_base_url`: 嵌入模型API基础URL
- `embedding_api_key`: 嵌入模型API密钥
- `embedding_batch_size`: 嵌入批处理大小，默认32

**检索相关：**
- `retrieval_top_k`: 检索的文档数量，默认10
- `qa_top_k`: 问答时使用的文档数量，默认5
- `damping`: PageRank阻尼因子，默认0.5
- `linking_top_k`: 链接时使用的top-k实体数量，默认10
- `passage_node_weight`: 段落节点权重，默认0.05

**图相关：**
- `graph_type`: 图类型，默认"facts_and_sim_passage_node_unidirectional"
- `is_directed_graph`: 是否为有向图，默认False
- `synonymy_edge_topk`: 同义词边的top-k值，默认10
- `synonymy_edge_sim_threshold`: 同义词边相似度阈值，默认0.8

**其他：**
- `save_dir`: 保存目录
- `save_openie`: 是否保存OpenIE结果，默认True
- `openie_mode`: OpenIE模式（仅支持'online'）
- `force_openie_from_scratch`: 是否强制重新运行OpenIE，默认False
- `force_index_from_scratch`: 是否强制重新索引，默认False

## 注意事项

1. **在线模式限制**：本版本仅支持在线模式，不支持离线索引模式

2. **API密钥配置**：API密钥需要在BaseConfig中配置，而不是通过环境变量设置

3. **中文支持**：提示词已翻译为中文，但评估指标（如`normalize_answer`）主要针对英文文本设计

4. **GPU使用**：如果使用GPU，请设置`CUDA_VISIBLE_DEVICES`环境变量

5. **保存目录**：每次使用不同的LLM/嵌入模型组合时，会在保存目录下创建新的子目录

## 论文引用

如果你觉得这项工作有用，请考虑引用我们的论文：

### HippoRAG 2

```bibtex
@misc{gutiérrez2025ragmemorynonparametriccontinual,
      title={From RAG to Memory: Non-Parametric Continual Learning for Large Language Models}, 
      author={Bernal Jiménez Gutiérrez and Yiheng Shu and Weijian Qi and Sizhe Zhou and Yu Su},
      year={2025},
      eprint={2502.14802},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.14802}, 
}
```

### HippoRAG

```bibtex
@inproceedings{gutiérrez2024hipporag,
      title={HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models}, 
      author={Bernal Jiménez Gutiérrez and Yiheng Shu and Yu Gu and Michihiro Yasunaga and Yu Su},
      booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
      year={2024},
      url={https://openreview.net/forum?id=hkujvAPVsg}
}
```

## 联系方式

有问题或建议？请提交issue或联系：
[Bernal Jiménez Gutiérrez](mailto:jimenezgutierrez.1@osu.edu),
[Yiheng Shu](mailto:shu.251@osu.edu),
[Yu Su](mailto:su.809@osu.edu),
俄亥俄州立大学

## 许可证

本项目基于原HippoRAG项目进行修改和简化。请参考原项目的许可证信息。

## 致谢

感谢原HippoRAG项目的作者和贡献者。

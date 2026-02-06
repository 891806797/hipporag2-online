# 评估模块 (Evaluation Module)

本目录包含用于评估HippoRAG检索增强生成（RAG）系统性能的评估指标实现。

## 概述

HippoRAG评估模块提供了多种评估指标，用于衡量系统在检索和问答任务中的表现。这些指标基于标准的QA评估方法，包括精确匹配（Exact Match）、F1分数和召回率（Recall）。

## 模块结构

```
hipporag/evaluation/
├── __init__.py           # 模块初始化文件
├── base.py               # 基础评估指标类
├── qa_eval.py            # 问答评估指标
├── retrieval_eval.py     # 检索评估指标
└── README.md             # 本文档
```

## 核心组件

### 1. 基础评估类 (`BaseMetric`)

位置：[`base.py`](base.py:14)

[`BaseMetric`](base.py:14)是所有评估指标的基类，提供了统一的接口和配置管理。

**主要属性：**
- `global_config`: 全局配置对象
- `metric_name`: 指标名称标识符

**主要方法：**
- `calculate_metric_scores()`: 计算评估指标的总分和每个样本的得分

### 2. 问答评估指标 (`qa_eval.py`)

位置：[`qa_eval.py`](qa_eval.py:1)

#### QAExactMatch（精确匹配）

[`QAExactMatch`](qa_eval.py:13)计算预测答案与标准答案的精确匹配率。

**使用方法：**

```python
from hipporag.evaluation.qa_eval import QAExactMatch

# 初始化评估器
qa_em_evaluator = QAExactMatch(global_config=global_config)

# 计算精确匹配分数
gold_answers = [["政治家"], ["通过参加舞会"], ["罗克兰县"]]
predicted_answers = ["政治家", "通过参加舞会", "罗克兰县"]

overall_result, example_results = qa_em_evaluator.calculate_metric_scores(
    gold_answers=gold_answers,
    predicted_answers=predicted_answers,
    aggregation_fn=np.max  # 使用最大值聚合多个标准答案
)

# overall_result: {"ExactMatch": 1.0}
# example_results: [{"ExactMatch": 1.0}, {"ExactMatch": 1.0}, {"ExactMatch": 1.0}]
```

**参数说明：**
- `gold_answers`: 标准答案列表（每个问题可能有多个标准答案）
- `predicted_answers`: 预测答案列表
- `aggregation_fn`: 聚合函数，默认为`np.max`，用于在多个标准答案中选择最佳匹配

#### QAF1Score（F1分数）

[`QAF1Score`](qa_eval.py:49)计算预测答案与标准答案之间的F1分数，综合考虑精确率和召回率。

**使用方法：**

```python
from hipporag.evaluation.qa_eval import QAF1Score

# 初始化评估器
qa_f1_evaluator = QAF1Score(global_config=global_config)

# 计算F1分数
overall_result, example_results = qa_f1_evaluator.calculate_metric_scores(
    gold_answers=gold_answers,
    predicted_answers=predicted_answers,
    aggregation_fn=np.max
)

# overall_result: {"F1": 0.95}
```

**F1分数计算原理：**
1. 将答案分词
2. 计算预测答案和标准答案之间的共同词数
3. 精确率 = 共同词数 / 预测答案词数
4. 召回率 = 共同词数 / 标准答案词数
5. F1 = 2 × (精确率 × 召回率) / (精确率 + 召回率)

### 3. 检索评估指标 (`retrieval_eval.py`)

位置：[`retrieval_eval.py`](retrieval_eval.py:1)

#### RetrievalRecall（召回率）

[`RetrievalRecall`](retrieval_eval.py:16)计算检索系统在不同k值下的召回率（Recall@k）。

**使用方法：**

```python
from hipporag.evaluation.retrieval_eval import RetrievalRecall

# 初始化评估器
retrieval_recall_evaluator = RetrievalRecall(global_config=global_config)

# 准备数据
gold_docs = [
    ["文档1", "文档2"],  # 问题1的标准文档
    ["文档3", "文档4", "文档5"],  # 问题2的标准文档
    ["文档6"]  # 问题3的标准文档
]

retrieved_docs = [
    ["文档1", "文档2", "文档10", "文档11"],  # 问题1的检索结果
    ["文档3", "文档10", "文档11", "文档12"],  # 问题2的检索结果
    ["文档10", "文档11", "文档12", "文档13"]  # 问题3的检索结果
]

# 计算Recall@k
k_list = [1, 5, 10, 20]
overall_result, example_results = retrieval_recall_evaluator.calculate_metric_scores(
    gold_docs=gold_docs,
    retrieved_docs=retrieved_docs,
    k_list=k_list
)

# overall_result: {"Recall@1": 0.3333, "Recall@5": 0.3333, "Recall@10": 0.3333, "Recall@20": 0.3333}
```

**参数说明：**
- `gold_docs`: 标准文档列表（每个问题对应一个标准文档集合）
- `retrieved_docs`: 检索结果列表（每个问题对应一个检索文档列表）
- `k_list`: 要计算的k值列表，默认为[1, 5, 10, 20]

**召回率计算原理：**
```
Recall@k = (前k个检索结果中的标准文档数) / (标准文档总数)
```

## 在HippoRAG中的集成

评估指标在HippoRAG的主要方法中被自动调用：

### 检索评估

在[`HippoRAG.retrieve()`](../HippoRAG.py:351)方法中，当提供`gold_docs`参数时会自动进行检索评估：

```python
# 检索并评估
retrieval_results, overall_retrieval_result = hipporag.retrieve(
    queries=queries,
    gold_docs=gold_docs  # 提供标准文档以启用评估
)

# overall_retrieval_result 包含各k值下的召回率
# {"Recall@1": 0.5, "Recall@5": 0.8, "Recall@10": 0.9, ...}
```

### 问答评估

在[`HippoRAG.rag_qa()`](../HippoRAG.py:439)方法中，当提供`gold_answers`参数时会自动进行问答评估：

```python
# 执行RAG问答并评估
query_solutions, responses, metadata, retrieval_result, qa_result = hipporag.rag_qa(
    queries=queries,
    gold_docs=gold_docs,      # 可选：用于检索评估
    gold_answers=gold_answers  # 用于问答评估
)

# qa_result 包含精确匹配和F1分数
# {"ExactMatch": 0.85, "F1": 0.92}
```

## 工具函数

### normalize_answer

位置：[`../utils/eval_utils.py`](../utils/eval_utils.py:4)

[`normalize_answer()`](../utils/eval_utils.py:4)函数用于标准化答案文本，包括以下步骤：

1. 转换为小写
2. 移除标点符号
3. 移除冠词（a, an, the）
4. 标准化空白字符

**使用示例：**

```python
from hipporag.utils.eval_utils import normalize_answer

answer = "The quick brown fox jumps over the lazy dog."
normalized = normalize_answer(answer)
# 结果: "quick brown fox jumps over lazy dog"
```

## 完整示例

以下是一个完整的评估流程示例：

```python
from hipporag import HippoRAG
from hipporag.evaluation.qa_eval import QAExactMatch, QAF1Score
from hipporag.evaluation.retrieval_eval import RetrievalRecall

# 初始化HippoRAG
hipporag = HippoRAG(
    save_dir='outputs',
    llm_model_name='gpt-4o-mini',
    embedding_model_name='nvidia/NV-Embed-v2'
)

# 准备文档
docs = [
    "Oliver Badman是一名政治家。",
    "George Rankin是一名政治家。",
    "Cinderella参加了皇家舞会。",
    "王子用丢失的水晶鞋搜寻王国。",
    "当鞋子完美合脚时，Cinderella与王子重聚。"
]

# 索引文档
hipporag.index(docs=docs)

# 准备查询和标准答案
queries = [
    "George Rankin的职业是什么？",
    "Cinderella是如何获得幸福结局的？"
]

gold_answers = [
    ["政治家"],
    ["通过参加舞会", "通过找到王子"]
]

gold_docs = [
    ["George Rankin是一名政治家。"],
    ["Cinderella参加了皇家舞会。", "王子用丢失的水晶鞋搜寻王国。", "当鞋子完美合脚时，Cinderella与王子重聚。"]
]

# 执行检索和问答（自动评估）
query_solutions, responses, metadata, retrieval_result, qa_result = hipporag.rag_qa(
    queries=queries,
    gold_docs=gold_docs,
    gold_answers=gold_answers
)

# 输出评估结果
print("检索评估结果:", retrieval_result)
print("问答评估结果:", qa_result)
```

## 注意事项

1. **答案标准化**：所有评估指标都使用`normalize_answer`函数对答案进行标准化，确保评估的公平性。

2. **多答案聚合**：当一个问题有多个标准答案时，使用`aggregation_fn`参数（默认为`np.max`）选择最佳匹配。

3. **k值选择**：检索评估支持多个k值，常用的k值包括1、5、10、20等。

4. **中文支持**：本项目的提示词已翻译为中文，但评估指标（如`normalize_answer`）主要针对英文文本设计。对于中文评估，可能需要调整标准化逻辑。

5. **在线模式**：本版本仅支持在线模式，所有评估均在实时检索和问答过程中进行。

## 扩展评估指标

如需添加自定义评估指标，可以继承[`BaseMetric`](base.py:14)类并实现`calculate_metric_scores`方法：

```python
from hipporag.evaluation.base import BaseMetric
from typing import List, Dict, Tuple, Union

class CustomMetric(BaseMetric):
    metric_name = "custom_metric"
    
    def calculate_metric_scores(self, **kwargs) -> Tuple[Dict[str, Union[int, float]], List[Union[int, float]]]:
        # 实现自定义评估逻辑
        scores = []
        for item in kwargs['items']:
            score = self._calculate_score(item)
            scores.append(score)
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        return {"CustomMetric": avg_score}, [{"CustomMetric": s} for s in scores]
```

## 参考文献

本评估模块的实现参考了以下标准QA评估方法：

- MRQA (Machine Reading for Question Answering) evaluation script
- SQuAD (Stanford Question Answering Dataset) evaluation metrics

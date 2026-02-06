import json
import os
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Union, Optional, List, Set, Dict, Any, Tuple, Literal
import numpy as np
from tqdm import tqdm
from igraph import Graph
import igraph as ig
from collections import defaultdict
import re
import time

from .llm import _get_llm_class, BaseLLM
from .embedding_model import _get_embedding_model_class, BaseEmbeddingModel
from .embedding_store import EmbeddingStore
from .information_extraction import OpenIE
from .evaluation.retrieval_eval import RetrievalRecall
from .evaluation.qa_eval import QAExactMatch, QAF1Score
from .prompts.linking import get_query_instruction
from .prompts.prompt_template_manager import PromptTemplateManager
from .utils.misc_utils import *
from .utils.embed_utils import retrieve_knn
from .utils.typing import Triple
from .utils.config_utils import BaseConfig

logger = logging.getLogger(__name__)

class StandardRAG:

    def __init__(self,
                 global_config=None,
                 save_dir=None,
                 llm_model_name=None,
                 embedding_model_name=None,
                 llm_base_url=None,
                 azure_endpoint=None,
                 azure_embedding_endpoint=None):
        """
        初始化StandardRAG实例及其相关组件。

        Attributes:
            global_config (BaseConfig): 实例的全局配置设置。如果未提供值，
                则使用BaseConfig实例。
            saving_dir (str): 存储特定StandardRAG实例的目录。如果未提供值，
                则默认为`outputs`。
            llm_model (BaseLLM): 基于全局配置设置用于处理的
                语言模型。
            embedding_model (BaseEmbeddingModel): 与当前配置关联的
                嵌入模型。
            chunk_embedding_store (EmbeddingStore): 处理块嵌入的嵌入存储。
            ready_to_retrieve (bool): 指示系统是否准备好进行检索操作的标志。

        Parameters:
            global_config: 全局配置对象。默认为None，导致初始化
                新的BaseConfig对象。
            working_dir: 存储工作文件的目录。默认为None，基于类名和时间戳
                构建默认目录。
            llm_model_name: LLM模型名称，可以直接插入，也可以通过配置文件插入。
            embedding_model_name: 嵌入模型名称，可以直接插入，也可以通过配置文件插入。
            llm_base_url: 部署的LLM模型的LLM URL，可以直接插入，也可以通过配置文件插入。
        """
        if global_config is None:
            self.global_config = BaseConfig()
        else:
            self.global_config = global_config

        # 如果指定，覆盖配置
        if save_dir is not None:
            self.global_config.save_dir = save_dir

        if llm_model_name is not None:
            self.global_config.llm_name = llm_model_name

        if embedding_model_name is not None:
            self.global_config.embedding_model_name = embedding_model_name

        if llm_base_url is not None:
            self.global_config.llm_base_url = llm_base_url

        if azure_endpoint is not None:
            self.global_config.azure_endpoint = azure_endpoint

        if azure_embedding_endpoint is not None:
            self.global_config.azure_embedding_endpoint = azure_embedding_endpoint

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self.global_config).items()])
        logger.debug(f"使用配置初始化StandardRAG:\n  {_print_config}\n")

        # 在每个指定的保存目录下创建LLM和嵌入模型特定的工作目录
        llm_label = self.global_config.llm_name.replace("/", "_")
        embedding_label = self.global_config.embedding_model_name.replace("/", "_")
        self.working_dir = os.path.join(self.global_config.save_dir, f"{llm_label}_{embedding_label}")

        if not os.path.exists(self.working_dir):
            logger.info(f"创建工作目录: {self.working_dir}")
            os.makedirs(self.working_dir, exist_ok=True)

        self.llm_model: BaseLLM = _get_llm_class(self.global_config)

        self.embedding_model: BaseEmbeddingModel = _get_embedding_model_class(
            embedding_model_name=self.global_config.embedding_model_name)(global_config=self.global_config,
                                                              embedding_model_name=self.global_config.embedding_model_name)

        self.chunk_embedding_store = EmbeddingStore(self.embedding_model,
                                                    os.path.join(self.working_dir, "chunk_embeddings"),
                                                    self.global_config.embedding_batch_size, 'chunk')

        self.ready_to_retrieve = False

        self.ppr_time = 0
        self.rerank_time = 0
        self.all_retrieval_time = 0

    def index(self, docs: List[str]):
        """
        基于HippoRAG 2框架索引给定的文档，该框架基于给定文档生成OpenIE知识图
        并分别编码段落、实体和事实以供后续检索。

        Parameters:
            docs : List[str]
                要索引的文档列表。
        """

        logger.info(f"索引文档")

        self.chunk_embedding_store.insert_strings(docs)

    def delete(self, docs_to_delete: List[str]):
        """
        从所有数据结构中删除给定文档。

        Parameters:
            docs : List[str]
                要删除的文档列表。
        """

        # 确保所有必要的结构都已构建
        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        current_docs = set(self.chunk_embedding_store.get_all_texts())
        docs_to_delete = [doc for doc in docs_to_delete if doc in current_docs]

        # 获取要删除的块的ID
        chunk_ids_to_delete = set(
            [self.chunk_embedding_store.text_to_hash_id[chunk] for chunk in docs_to_delete])

        logger.info(f"删除{len(chunk_ids_to_delete)}个块")

        self.chunk_embedding_store.delete(chunk_ids_to_delete)

        self.ready_to_retrieve = False

    def retrieve(self,
                 queries: List[str],
                 num_to_retrieve: int = None,
                 gold_docs: List[List[str]] = None) -> List[QuerySolution] | Tuple[List[QuerySolution], Dict]:
        """
        使用DPR框架执行检索，包括以下几个步骤：
        - 密集段落评分

        Parameters:
            queries: List[str]
                要检索文档的查询字符串列表。
            num_to_retrieve: int, optional
                为每个查询检索的最大文档数。如果未指定，默认为
                全局配置中定义的`retrieval_top_k`值。
            gold_docs: List[List[str]], optional
                包含与每个查询对应的标准文档的列表。如果启用了
                检索性能评估（全局配置中的`do_eval_retrieval`），则必需。

        Returns:
            List[QuerySolution] or (List[QuerySolution], Dict)
                如果未启用检索性能评估，返回QuerySolution对象列表，每个对象包含
                相应查询的检索文档及其分数。如果启用了评估，还返回
                包含对检索结果计算的评估指标的字典。

        Notes
        -----
        - 重排序后没有相关事实的长查询将默认为密集段落检索的结果。
        """
        retrieve_start_time = time.time()  # 记录开始时间

        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k

        if gold_docs is not None:
            retrieval_recall_evaluator = RetrievalRecall(global_config=self.global_config)

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        self.get_query_embeddings(queries)

        retrieval_results = []

        for q_idx, query in tqdm(enumerate(queries), desc="检索", total=len(queries)):
            logger.info('重排序后未发现事实，返回DPR结果')
            sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)

            top_k_docs = [self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"] for idx in
                          sorted_doc_ids[:num_to_retrieve]]

            retrieval_results.append(
                QuerySolution(question=query, docs=top_k_docs, doc_scores=sorted_doc_scores[:num_to_retrieve]))

        retrieve_end_time = time.time()  # 记录结束时间

        self.all_retrieval_time += retrieve_end_time - retrieve_start_time

        logger.info(f"总检索时间 {self.all_retrieval_time:.2f}s")

        # 评估检索
        if gold_docs is not None:
            k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
            overall_retrieval_result, example_retrieval_results = retrieval_recall_evaluator.calculate_metric_scores(
                gold_docs=gold_docs, retrieved_docs=[retrieval_result.docs for retrieval_result in retrieval_results],
                k_list=k_list)
            logger.info(f"检索的评估结果: {overall_retrieval_result}")

            return retrieval_results, overall_retrieval_result
        else:
            return retrieval_results

    def rag_qa(self,
               queries: List[str|QuerySolution],
               gold_docs: List[List[str]] = None,
               gold_answers: List[List[str]] = None) -> Tuple[List[QuerySolution], List[str], List[Dict]] | Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]:
        """
        使用标准DPR框架执行检索增强生成增强的QA。

        此方法可以处理基于字符串的查询和预处理的QuerySolution对象。根据
        其输入，它仅返回答案或另外使用
        召回@k、精确匹配和F1分数指标评估检索和答案质量。

        Parameters:
            queries (List[Union[str, QuerySolution]]): 查询列表，可以是字符串或
                QuerySolution实例。如果是字符串，将执行检索。
            gold_docs (Optional[List[List[str]]): 包含每个查询的标准文档的列表。
                如果要执行文档级评估，则使用此项。默认为None。
            gold_answers (Optional[List[List[str]]): 包含每个查询的标准答案的列表。
                如果启用了问题回答（QA）答案的评估，则必需此项。默认
                为None。

        Returns:
            Union[
                Tuple[List[QuerySolution], List[str], List[Dict]],
                Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]
            ]: 始终包含以下内容的元组：
                - 包含每个查询的答案和元数据的QuerySolution对象列表。
                - 为提供的查询提供的响应消息列表。
                - 与每个结果关联的元数据字典列表。
                如果启用了评估，该元组还包括：
                - 检索阶段的总体结果字典（如果适用）。
                - QA评估指标（精确匹配和F1分数）的总体字典。

        """
        if gold_answers is not None:
            qa_em_evaluator = QAExactMatch(global_config=self.global_config)
            qa_f1_evaluator = QAF1Score(global_config=self.global_config)

        # 检索（如果需要）
        overall_retrieval_result = None

        if not isinstance(queries[0], QuerySolution):
            if gold_docs is not None:
                queries, overall_retrieval_result = self.retrieve(queries=queries, gold_docs=gold_docs)
            else:
                queries = self.retrieve_dpr(queries=queries)

        # 执行QA
        queries_solutions, all_response_message, all_metadata = self.qa(queries)

        # 评估QA
        if gold_answers is not None:
            overall_qa_em_result, example_qa_em_results = qa_em_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)
            overall_qa_f1_result, example_qa_f1_results = qa_f1_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)

            # 将QA结果四舍五入到4位小数
            overall_qa_em_result.update(overall_qa_f1_result)
            overall_qa_results = overall_qa_em_result
            overall_qa_results = {k: round(float(v), 4) for k, v in overall_qa_results.items()}
            logger.info(f"QA的评估结果: {overall_qa_results}")

            # 保存检索和QA结果
            for idx, q in enumerate(queries_solutions):
                q.gold_answers = list(gold_answers[idx])
                if gold_docs is not None:
                    q.gold_docs = gold_docs[idx]

            return queries_solutions, all_response_message, all_metadata, overall_retrieval_result, overall_qa_results
        else:
            return queries_solutions, all_response_message, all_metadata

    def qa(self, queries: List[QuerySolution]) -> Tuple[List[QuerySolution], List[str], List[Dict]]:
        """
        使用提供的查询解决方案集和语言模型执行问答（QA）推理。

        Parameters:
            queries: List[QuerySolution]
                包含用户查询、检索文档和其他相关信息的QuerySolution对象列表。

        Returns:
            Tuple[List[QuerySolution], List[str], List[Dict]]
                包含以下内容的元组：
                - 包含预测答案的更新QuerySolution对象列表。
                - 来自语言模型的原始响应消息列表。
                - 与结果关联的元数据字典列表。
        """
        # 运行QA推理
        all_qa_messages = []

        for query_solution in tqdm(queries, desc="收集QA提示词"):

            # 获取检索到的文档
            retrieved_passages = query_solution.docs[:self.global_config.qa_top_k]

            prompt_user = ''
            for passage in retrieved_passages:
                prompt_user += f'维基百科标题: {passage}\n\n'
            prompt_user += '问题: ' + query_solution.question + '\n思考: '

            if self.prompt_template_manager.is_template_name_valid(name=f'rag_qa_{self.global_config.dataset}'):
                # 找到该数据集对应的提示词
                prompt_dataset_name = self.global_config.dataset
            else:
                # 该数据集还没有自定义提示词模板
                logger.debug(
                    f"rag_qa_{self.global_config.dataset}没有自定义提示词模板。使用MUSIQUE的提示词模板代替。")
                prompt_dataset_name = 'musique'
            all_qa_messages.append(
                self.prompt_template_manager.render(name=f'rag_qa_{prompt_dataset_name}', prompt_user=prompt_user))

        all_qa_results = [self.llm_model.infer(qa_messages) for qa_messages in tqdm(all_qa_messages, desc="QA读取")]

        all_response_message, all_metadata, all_cache_hit = zip(*all_qa_results)
        all_response_message, all_metadata = list(all_response_message), list(all_metadata)

        # 处理响应并从LLM响应中提取预测答案
        queries_solutions = []
        for query_solution_idx, query_solution in tqdm(enumerate(queries), desc="从LLM响应中提取答案"):
            response_content = all_response_message[query_solution_idx]
            try:
                pred_ans = response_content.split('答案:')[1].strip()
            except Exception as e:
                logger.warning(f"从原始LLM QA推理响应中解析答案时出错: {str(e)}!")
                pred_ans = response_content

            query_solution.answer = pred_ans
            queries_solutions.append(query_solution)

        return queries_solutions, all_response_message, all_metadata


    def prepare_retrieval_objects(self):
        """
        准备快速检索过程所需的各种内存对象和属性，例如嵌入数据和图关系，确保一致性
        并与底层图结构对齐。
        """

        logger.info("准备快速检索。")

        logger.info("加载键。")
        self.query_to_embedding: Dict = {'triple': {}, 'passage': {}}

        self.passage_node_keys: List = list(self.chunk_embedding_store.get_all_ids()) # 段落节点键列表

        logger.info("加载嵌入。")
        self.passage_embeddings = np.array(self.chunk_embedding_store.get_embeddings(self.passage_node_keys))

        self.ready_to_retrieve = True

    def get_query_embeddings(self, queries: List[str] | List[QuerySolution]):
        """
        检索给定查询的嵌入并更新内部查询到嵌入的映射。该方法确定每个查询
        是否已经存在于`self.query_to_embedding`字典中的'triple'和'passage'键下。如果查询不存在于
        任一位置，它将使用嵌入模型进行编码并存储。

        Args:
            queries List[str] | List[QuerySolution]: 查询字符串或QuerySolution对象的列表。检查每个查询
            其在查询到嵌入映射中的存在性。
        """

        all_query_strings = []
        for query in queries:
            if isinstance(query, QuerySolution) and (
                    query.question not in self.query_to_embedding['triple'] or query.question not in
                    self.query_to_embedding['passage']):
                all_query_strings.append(query.question)
            elif query not in self.query_to_embedding['triple'] or query not in self.query_to_embedding['passage']:
                all_query_strings.append(query)

        if len(all_query_strings) > 0:
            logger.info(f"为query_to_passage编码{len(all_query_strings)}个查询。")
            query_embeddings_for_passage = self.embedding_model.batch_encode(all_query_strings,
                                                                             instruction=get_query_instruction('query_to_passage'),
                                                                             norm=True)
            for query, embedding in zip(all_query_strings, query_embeddings_for_passage):
                self.query_to_embedding['passage'][query] = embedding

    def dense_passage_retrieval(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行密集段落检索以查找查询的相关文档。

        此函数使用预训练的嵌入模型处理给定查询
        以生成查询嵌入。使用点积计算查询嵌入与段落嵌入之间的相似度分数，然后
        进行分数归一化。最后，函数基于
        相似度分数对文档进行排序，并返回排序后的文档标识符
        及其分数。

        Parameters
        ----------
        query : str
            应该为其检索相关段落的输入查询。

        Returns
        -------
        tuple : Tuple[np.ndarray, np.ndarray]
            包含两个元素的元组：
            - 基于其相关性分数排序的文档标识符列表。
            - 相应文档的归一化相似度分数的numpy数组。
        """
        query_embedding = self.query_to_embedding['passage'].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(query,
                                                                instruction=get_query_instruction('query_to_passage'),
                                                                norm=True)
        query_doc_scores = np.dot(self.passage_embeddings, query_embedding.T)
        query_doc_scores = np.squeeze(query_doc_scores) if query_doc_scores.ndim == 2 else query_doc_scores
        query_doc_scores = min_max_normalize(query_doc_scores)

        sorted_doc_ids = np.argsort(query_doc_scores)[::-1]
        sorted_doc_scores = query_doc_scores[sorted_doc_ids.tolist()]

        return sorted_doc_ids, sorted_doc_scores

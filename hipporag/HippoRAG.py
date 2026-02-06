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
from .rerank import DSPyFilter
from .utils.misc_utils import *
from .utils.misc_utils import NerRawOutput, TripleRawOutput
from .utils.embed_utils import retrieve_knn
from .utils.typing import Triple
from .utils.config_utils import BaseConfig

logger = logging.getLogger(__name__)

class HippoRAG:

    def __init__(self,
                 global_config=None,
                 save_dir=None,
                 llm_model_name=None,
                 llm_base_url=None,
                 embedding_model_name=None,
                 embedding_base_url=None,
                 azure_endpoint=None,
                 azure_embedding_endpoint=None):
        """
        初始化HippoRAG实例及其相关组件。

        Attributes:
            global_config (BaseConfig): 实例的全局配置设置。如果未提供值，
                则使用BaseConfig实例。
            saving_dir (str): 存储特定HippoRAG实例的目录。如果未提供值，
                则默认为`outputs`。
            llm_model (BaseLLM): 基于全局配置设置用于处理的
                语言模型。
            openie (OpenIE): 配置为在线模式的开放信息提取模块。
            graph: 由`initialize_graph`方法初始化的图实例。
            embedding_model (BaseEmbeddingModel): 与当前配置关联的
                嵌入模型。
            chunk_embedding_store (EmbeddingStore): 处理块嵌入的嵌入存储。
            entity_embedding_store (EmbeddingStore): 处理实体嵌入的嵌入存储。
            fact_embedding_store (EmbeddingStore): 处理事实嵌入的嵌入存储。
            prompt_template_manager (PromptTemplateManager): 处理提示词模板和
                角色映射的管理器。
            openie_results_path (str): 基于数据集和全局配置中的LLM名称
                存储开放信息提取结果的文件路径。
            rerank_filter (Optional[DSPyFilter]): 当在全局配置中指定重排序文件路径时，
                负责重排序信息的过滤器。
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

        if embedding_base_url is not None:
            self.global_config.embedding_base_url = embedding_base_url

        if azure_endpoint is not None:
            self.global_config.azure_endpoint = azure_endpoint

        if azure_embedding_endpoint is not None:
            self.global_config.azure_embedding_endpoint = azure_embedding_endpoint

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self.global_config).items()])
        logger.debug(f"使用配置初始化HippoRAG:\n  {_print_config}\n")

        # 在每个指定的保存目录下创建LLM和嵌入模型特定的工作目录
        llm_label = self.global_config.llm_name.replace("/", "_")
        embedding_label = self.global_config.embedding_model_name.replace("/", "_")
        self.working_dir = os.path.join(self.global_config.save_dir, f"{llm_label}_{embedding_label}")

        if not os.path.exists(self.working_dir):
            logger.info(f"创建工作目录: {self.working_dir}")
            os.makedirs(self.working_dir, exist_ok=True)

        self.llm_model: BaseLLM = _get_llm_class(self.global_config)

        if self.global_config.openie_mode == 'online':
            self.openie = OpenIE(llm_model=self.llm_model)
        else:
            raise ValueError(f"仅支持在线模式，不支持离线模式: {self.global_config.openie_mode}")

        self.graph = self.initialize_graph()

        self.embedding_model: BaseEmbeddingModel = _get_embedding_model_class(
            embedding_model_name=self.global_config.embedding_model_name)(global_config=self.global_config,
                                                              embedding_model_name=self.global_config.embedding_model_name)
        self.chunk_embedding_store = EmbeddingStore(self.embedding_model,
                                                    os.path.join(self.working_dir, "chunk_embeddings"),
                                                    self.global_config.embedding_batch_size, 'chunk')
        self.entity_embedding_store = EmbeddingStore(self.embedding_model,
                                                     os.path.join(self.working_dir, "entity_embeddings"),
                                                     self.global_config.embedding_batch_size, 'entity')
        self.fact_embedding_store = EmbeddingStore(self.embedding_model,
                                                   os.path.join(self.working_dir, "fact_embeddings"),
                                                   self.global_config.embedding_batch_size, 'fact')

        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})

        self.openie_results_path = os.path.join(self.global_config.save_dir,f'openie_results_ner_{self.global_config.llm_name.replace("/", "_")}.json')

        self.rerank_filter = DSPyFilter(self)

        self.ready_to_retrieve = False

        self.ppr_time = 0
        self.rerank_time = 0
        self.all_retrieval_time = 0

        self.ent_node_to_chunk_ids = None


    def initialize_graph(self):
        """
        如果可用，使用Pickle文件初始化图，或创建新图。

        该函数尝试加载存储在Pickle文件中的预存图。如果文件
        不存在或需要从头创建图，它将基于全局配置初始化新的有向
        或无向图。如果成功从文件加载图，则记录有关图的
        信息（节点数和边数）。

        Returns:
            ig.Graph: 预加载或新初始化的图。

        Raises:
            None
        """
        self._graph_pickle_filename = os.path.join(
            self.working_dir, f"graph.pickle"
        )

        preloaded_graph = None

        if not self.global_config.force_index_from_scratch:
            if os.path.exists(self._graph_pickle_filename):
                preloaded_graph = ig.Graph.Read_Pickle(self._graph_pickle_filename)

        if preloaded_graph is None:
            return ig.Graph(directed=self.global_config.is_directed_graph)
        else:
            logger.info(
                f"从{self._graph_pickle_filename}加载图，包含{preloaded_graph.vcount()}个节点，{preloaded_graph.ecount()}条边"
            )
            return preloaded_graph

    def pre_openie(self,  docs: List[str]):
        """
        预处理OpenIE（开放信息提取）。

        Args:
            docs (List[str]): 要索引的文档列表。
        """
        logger.info(f"索引文档")
        logger.info(f"执行OpenIE离线")

        chunks = self.chunk_embedding_store.get_missing_string_hash_ids(docs)

        all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunks.keys())
        new_openie_rows = {k : chunks[k] for k in chunk_keys_to_process}

        if len(chunk_keys_to_process) > 0:
            new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(new_openie_rows)
            self.merge_openie_results(all_openie_info, new_openie_rows, new_ner_results_dict, new_triple_results_dict)

        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)

        assert False, logger.info('OpenIE完成，运行在线索引以进行后续检索。')

    def index(self, docs: List[str]):
        """
        基于HippoRAG 2框架索引给定的文档，该框架基于给定文档生成OpenIE知识图
        并分别编码段落、实体和事实以供后续检索。

        Parameters:
            docs : List[str]
                要索引的文档列表。
        """

        logger.info(f"索引文档")

        logger.info(f"执行OpenIE")

        self.chunk_embedding_store.insert_strings(docs)
        chunk_to_rows = self.chunk_embedding_store.get_all_id_to_rows()

        all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunk_to_rows.keys())
        new_openie_rows = {k : chunk_to_rows[k] for k in chunk_keys_to_process}

        if len(chunk_keys_to_process) > 0:
            new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(new_openie_rows)
            self.merge_openie_results(all_openie_info, new_openie_rows, new_ner_results_dict, new_triple_results_dict)

        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)

        ner_results_dict, triple_results_dict = reformat_openie_results(all_openie_info)

        assert len(chunk_to_rows) == len(ner_results_dict) == len(triple_results_dict), f"len(chunk_to_rows): {len(chunk_to_rows)}, len(ner_results_dict): {len(ner_results_dict)}, len(triple_results_dict): {len(triple_results_dict)}"

        # 准备data_store
        chunk_ids = list(chunk_to_rows.keys())

        chunk_triples = [[text_processing(t) for t in triple_results_dict[chunk_id].triples] for chunk_id in chunk_ids]
        entity_nodes, chunk_triple_entities = extract_entity_nodes(chunk_triples)
        facts = flatten_facts(chunk_triples)

        logger.info(f"编码实体")
        self.entity_embedding_store.insert_strings(entity_nodes)

        logger.info(f"编码事实")
        self.fact_embedding_store.insert_strings([str(fact) for fact in facts])

        logger.info(f"构建图")

        self.node_to_node_stats = {}
        self.ent_node_to_chunk_ids = {}

        self.add_fact_edges(chunk_ids, chunk_triples)
        num_new_chunks = self.add_passage_edges(chunk_ids, chunk_triple_entities)

        if num_new_chunks > 0:
            logger.info(f"发现{num_new_chunks}个新块要保存到图中。")
            self.add_synonymy_edges()

            self.augment_graph()
            self.save_igraph()

    def delete(self, docs_to_delete: List[str]):
        """
        从HippoRAG类中的所有数据结构中删除给定文档。注意，从未被删除的块中索引的三元组和实体将不会被删除。

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

        # 在要删除的块中查找三元组
        all_openie_info, chunk_keys_to_process = self.load_existing_openie([])
        triples_to_delete = []

        all_openie_info_with_deletes = []

        for openie_doc in all_openie_info:
            if openie_doc['idx'] in chunk_ids_to_delete:
                triples_to_delete.append(openie_doc['extracted_triples'])
            else:
                all_openie_info_with_deletes.append(openie_doc)

        triples_to_delete = flatten_facts(triples_to_delete)

        # 过滤出出现在未更改块中的三元组
        true_triples_to_delete = []

        for triple in triples_to_delete:
            proc_triple = tuple(text_processing(list(triple)))

            doc_ids = self.proc_triples_to_docs[str(proc_triple)]

            non_deleted_docs = doc_ids.difference(chunk_ids_to_delete)

            if len(non_deleted_docs) == 0:
                true_triples_to_delete.append(triple)

        processed_true_triples_to_delete = [[text_processing(list(triple)) for triple in true_triples_to_delete]]
        entities_to_delete, _ = extract_entity_nodes(processed_true_triples_to_delete)
        processed_true_triples_to_delete = flatten_facts(processed_true_triples_to_delete)

        triple_ids_to_delete = set([self.fact_embedding_store.text_to_hash_id[str(triple)] for triple in processed_true_triples_to_delete])

        # 过滤出出现在未更改块中的实体
        ent_ids_to_delete = [self.entity_embedding_store.text_to_hash_id[ent] for ent in entities_to_delete]

        filtered_ent_ids_to_delete = []

        for ent_node in ent_ids_to_delete:
            doc_ids = self.ent_node_to_chunk_ids[ent_node]

            non_deleted_docs = doc_ids.difference(chunk_ids_to_delete)

            if len(non_deleted_docs) == 0:
                filtered_ent_ids_to_delete.append(ent_node)

        logger.info(f"删除{len(chunk_ids_to_delete)}个块")
        logger.info(f"删除{len(triple_ids_to_delete)}个三元组")
        logger.info(f"删除{len(filtered_ent_ids_to_delete)}个实体")

        self.save_openie_results(all_openie_info_with_deletes)

        self.entity_embedding_store.delete(filtered_ent_ids_to_delete)
        self.fact_embedding_store.delete(triple_ids_to_delete)
        self.chunk_embedding_store.delete(chunk_ids_to_delete)

        # 从图中删除节点
        self.graph.delete_vertices(list(filtered_ent_ids_to_delete) + list(chunk_ids_to_delete))
        self.save_igraph()

        self.ready_to_retrieve = False

    def retrieve(self,
                 queries: List[str],
                 num_to_retrieve: int = None,
                 gold_docs: List[List[str]] = None) -> List[QuerySolution] | Tuple[List[QuerySolution], Dict]:
        """
        使用HippoRAG 2框架执行检索，包括以下几个步骤：
        - 事实检索
        - 用于改进事实选择的识别记忆
        - 密集段落评分
        - 基于个性化PageRank的重排序

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
            rerank_start = time.time()
            query_fact_scores = self.get_fact_scores(query)
            top_k_fact_indices, top_k_facts, rerank_log = self.rerank_facts(query, query_fact_scores)
            rerank_end = time.time()

            self.rerank_time += rerank_end - rerank_start

            if len(top_k_facts) == 0:
                logger.info('重排序后未发现事实，返回DPR结果')
                sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)
            else:
                sorted_doc_ids, sorted_doc_scores = self.graph_search_with_fact_entities(query=query,
                                                                                         link_top_k=self.global_config.linking_top_k,
                                                                                         query_fact_scores=query_fact_scores,
                                                                                         top_k_facts=top_k_facts,
                                                                                         top_k_fact_indices=top_k_fact_indices,
                                                                                         passage_node_weight=self.global_config.passage_node_weight)

            top_k_docs = [self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"] for idx in sorted_doc_ids[:num_to_retrieve]]

            retrieval_results.append(QuerySolution(question=query, docs=top_k_docs, doc_scores=sorted_doc_scores[:num_to_retrieve]))

        retrieve_end_time = time.time()  # 记录结束时间

        self.all_retrieval_time += retrieve_end_time - retrieve_start_time

        logger.info(f"总检索时间 {self.all_retrieval_time:.2f}s")
        logger.info(f"总识别记忆时间 {self.rerank_time:.2f}s")
        logger.info(f"总PPR时间 {self.ppr_time:.2f}s")
        logger.info(f"总杂项时间 {self.all_retrieval_time - (self.rerank_time + self.ppr_time):.2f}s")

        # 评估检索
        if gold_docs is not None:
            k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
            overall_retrieval_result, example_retrieval_results = retrieval_recall_evaluator.calculate_metric_scores(gold_docs=gold_docs, retrieved_docs=[retrieval_result.docs for retrieval_result in retrieval_results], k_list=k_list)
            logger.info(f"检索的评估结果: {overall_retrieval_result}")

            return retrieval_results, overall_retrieval_result
        else:
            return retrieval_results

    def rag_qa(self,
               queries: List[str|QuerySolution],
               gold_docs: List[List[str]] = None,
               gold_answers: List[List[str]] = None) -> Tuple[List[QuerySolution], List[str], List[Dict]] | Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]:
        """
        使用HippoRAG 2框架执行检索增强生成增强的QA。

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
                queries = self.retrieve(queries=queries)

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

    def retrieve_dpr(self,
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

    def rag_qa_dpr(self,
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
                queries, overall_retrieval_result = self.retrieve_dpr(queries=queries, gold_docs=gold_docs)
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

    def add_fact_edges(self, chunk_ids: List[str], chunk_triples: List[Tuple]):
        """
        将事实边从给定的三元组添加到图中。

        该方法处理三元组块，计算实体和关系的唯一标识符，
        并更新各种内部统计信息以构建和维护图结构。实体被唯一
        标识并根据它们的关系进行链接。

        Parameters:
            chunk_ids: List[str]
                正在处理的块的唯一标识符列表。
            chunk_triples: List[Tuple]
                要处理的三元组列表。每个三元组
                由主语、谓语和宾语组成。

        Raises:
            在提供的函数逻辑中不会显式引发异常。
        """

        if "name" in self.graph.vs:
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        logger.info(f"将OpenIE三元组添加到图中。")

        for chunk_key, triples in tqdm(zip(chunk_ids, chunk_triples)):
            entities_in_chunk = set()

            if chunk_key not in current_graph_nodes:
                for triple in triples:
                    triple = tuple(triple)

                    node_key = compute_mdhash_id(content=triple[0], prefix=("entity-"))
                    node_2_key = compute_mdhash_id(content=triple[2], prefix=("entity-"))

                    self.node_to_node_stats[(node_key, node_2_key)] = self.node_to_node_stats.get(
                        (node_key, node_2_key), 0.0) + 1
                    self.node_to_node_stats[(node_2_key, node_key)] = self.node_to_node_stats.get(
                        (node_2_key, node_key), 0.0) + 1

                    entities_in_chunk.add(node_key)
                    entities_in_chunk.add(node_2_key)

                for node in entities_in_chunk:
                    self.ent_node_to_chunk_ids[node] = self.ent_node_to_chunk_ids.get(node, set()).union(set([chunk_key]))

    def add_passage_edges(self, chunk_ids: List[str], chunk_triple_entities: List[List[str]]):
        """
        在图中添加连接段落节点到短语节点的边。

        此方法负责迭代块标识符列表
        及其对应的三元组实体。它计算并添加新边
        在段落节点（由块标识符定义）和短语
        节点（由三元组实体的计算唯一哈希ID定义）之间。该方法
        还更新节点到节点统计映射并保持新添加的
        段落节点的计数。

        Parameters:
            chunk_ids : List[str]
                表示图中段落节点的标识符列表。
            chunk_triple_entities : List[List[str]]
                列表的列表，其中每个子列表包含与
                chunk_ids列表中对应的块关联的实体（字符串）。

        Returns:
            int
                添加到图中的新段落节点数。
        """

        if "name" in self.graph.vs.attribute_names():
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        num_new_chunks = 0

        logger.info(f"连接段落节点到短语节点。")

        for idx, chunk_key in tqdm(enumerate(chunk_ids)):

            if chunk_key not in current_graph_nodes:
                for chunk_ent in chunk_triple_entities[idx]:
                    node_key = compute_mdhash_id(chunk_ent, prefix="entity-")

                    self.node_to_node_stats[(chunk_key, node_key)] = 1.0

                num_new_chunks += 1

        return num_new_chunks

    def add_synonymy_edges(self):
        """
        通过识别和链接同义实体，在相似节点之间添加同义词边以增强图连接性。

        此方法执行关键操作以计算和添加同义词边。它首先检索所有节点的嵌入，
        然后执行最近邻（KNN）搜索以查找相似节点。这些相似节点基于分数阈值进行识别，
        并添加边以表示同义词关系。

        Attributes:
            entity_id_to_row: dict（在函数内部填充）。将每个实体ID映射到其对应的行数据，其中行
                              包含用于比较的实体的`content`。
            entity_embedding_store: 管理与实体相关的所有行的文本和嵌入检索。
            global_config: 定义诸如`synonymy_edge_topk`、`synonymy_edge_sim_threshold`、
                           `synonymy_edge_query_batch_size`和`synonymy_edge_key_batch_size`等参数的配置对象。
            node_to_node_stats: dict。存储表示节点之间关系的边的分数。
        """
        logger.info(f"用同义词边扩展图")

        self.entity_id_to_row = self.entity_embedding_store.get_all_id_to_rows()
        entity_node_keys = list(self.entity_id_to_row.keys())

        logger.info(f"为每个短语节点（{len(entity_node_keys)}）执行KNN检索。")

        entity_embs = self.entity_embedding_store.get_embeddings(entity_node_keys)

        # 这里我们仅在插入的新短语节点和存储中的所有短语节点之间构建同义词边，以减少增量图更新的成本
        query_node_key2knn_node_keys = retrieve_knn(query_ids=entity_node_keys,
                                                    key_ids=entity_node_keys,
                                                    query_vecs=entity_embs,
                                                    key_vecs=entity_embs,
                                                    k=self.global_config.synonymy_edge_topk,
                                                    query_batch_size=self.global_config.synonymy_edge_query_batch_size,
                                                    key_batch_size=self.global_config.synonymy_edge_key_batch_size)

        num_synonym_triple = 0
        synonym_candidates = []  # [(node key, [(synonym node key, corresponding score), ...]), ...]

        for node_key in tqdm(query_node_key2knn_node_keys.keys(), total=len(query_node_key2knn_node_keys)):
            synonyms = []

            entity = self.entity_id_to_row[node_key]["content"]

            if len(re.sub('[^A-Za-z0-9]', '', entity)) > 2:
                nns = query_node_key2knn_node_keys[node_key]

                num_nns = 0
                for nn, score in zip(nns[0], nns[1]):
                    if score < self.global_config.synonymy_edge_sim_threshold or num_nns > 100:
                        break

                    nn_phrase = self.entity_id_to_row[nn]["content"]

                    if nn != node_key and nn_phrase != '':
                        sim_edge = (node_key, nn)
                        synonyms.append((nn, score))
                        num_synonym_triple += 1

                        self.node_to_node_stats[sim_edge] = score  # 需要认真讨论这一点
                        num_nns += 1

            synonym_candidates.append((node_key, synonyms))

    def load_existing_openie(self, chunk_keys: List[str]) -> Tuple[List[dict], Set[str]]:
        """
        如果文件存在，从指定文件加载现有的OpenIE结果，并与新内容合并，
        同时标准化索引。如果文件不存在或使用标志`force_openie_from_scratch`
        配置为重新初始化，它准备要处理的新条目。

        Args:
            chunk_keys (List[str]): 表示要处理的内容的
                                      块键列表。

        Returns:
            Tuple[List[dict], Set[str]]: 第一个元素是现有的OpenIE
                                         信息（如果有）从文件加载的，第二个
                                         元素是仍需要保存或处理的块键集合。
        """

        # 将openie_results与文件中已存在的内容合并（如果文件存在）
        chunk_keys_to_save = set()

        if not self.global_config.force_openie_from_scratch and os.path.isfile(self.openie_results_path):
            openie_results = json.load(open(self.openie_results_path))
            all_openie_info = openie_results.get('docs', [])

            # 标准化OpenIE文件的索引

            renamed_openie_info = []
            for openie_info in all_openie_info:
                openie_info['idx'] = compute_mdhash_id(openie_info['passage'], 'chunk-')
                renamed_openie_info.append(openie_info)

            all_openie_info = renamed_openie_info

            existing_openie_keys = set([info['idx'] for info in all_openie_info])

            for chunk_key in chunk_keys:
                if chunk_key not in existing_openie_keys:
                    chunk_keys_to_save.add(chunk_key)
        else:
            all_openie_info = []
            chunk_keys_to_save = chunk_keys

        return all_openie_info, chunk_keys_to_save

    def merge_openie_results(self,
                             all_openie_info: List[dict],
                             chunks_to_save: Dict[str, dict],
                             ner_results_dict: Dict[str, NerRawOutput],
                             triple_results_dict: Dict[str, TripleRawOutput]) -> List[dict]:
        """
        将OpenIE提取结果与相应的段落和元数据合并。

        此函数集成OpenIE提取结果，包括命名实体
        识别（NER）实体和三元组，使用提供的块键与它们各自的文本段落合并。
        生成的合并数据被追加到
        `all_openie_info`列表中，其中包含用于进一步处理或存储的组合和组织
        数据。

        Parameters:
            all_openie_info (List[dict]): 包含所有块的合并OpenAI结果的列表
                结果和元数据。
            chunks_to_save (Dict[str, dict]): 要处理的块标识符（键）的字典
                以及合并OpenIE结果到具有`hash_id`和`content`键的字典。
            ner_results_dict (Dict[str, NerRawOutput]): 将块键
                映射到其对应的NER提取结果的字典。
            triple_results_dict (Dict[str, TripleRawOutput]): 将块
                键映射到其对应的OpenAI三元组提取结果的字典。

        Returns:
            List[dict]: `all_openie_info`列表，包含合并的
            OpenAI结果、元数据和每个块的段落内容。
        """

        for chunk_key, row in chunks_to_save.items():
            passage = row['content']
            try:
                chunk_openie_info = {'idx': chunk_key, 'passage': passage,
                                 'extracted_entities': ner_results_dict[chunk_key].unique_entities,
                                 'extracted_triples': triple_results_dict[chunk_key].triples}
            except Exception as e:
                logger.error(f"处理块{chunk_key}时出错: {e}")
                chunk_openie_info = {'idx': chunk_key, 'passage': passage,
                                 'extracted_entities': [],
                                 'extracted_triples': []}
            all_openie_info.append(chunk_openie_info)

        return all_openie_info

    def save_openie_results(self, all_openie_info: List[dict]):
        """
        计算从OpenAI结果中提取的实体的统计信息，并将聚合数据保存在JSON文件中。
        该函数计算提取实体的平均字符和单词长度，并将它们与提供的OpenAI信息一起写入文件。

        Parameters:
            all_openie_info : List[dict]
                字典列表，其中每个字典表示来自OpenAI的信息，包括
                提取的实体。
        """

        sum_phrase_chars = sum([len(e) for chunk in all_openie_info for e in chunk['extracted_entities']])
        sum_phrase_words = sum([len(e.split()) for chunk in all_openie_info for e in chunk['extracted_entities']])
        num_phrases = sum([len(chunk['extracted_entities']) for chunk in all_openie_info])

        if len(all_openie_info) > 0:
            # 避免除以零，如果没有短语
            if num_phrases > 0:
                avg_ent_chars = round(sum_phrase_chars / num_phrases, 4)
                avg_ent_words = round(sum_phrase_words / num_phrases, 4)
            else:
                avg_ent_chars = 0
                avg_ent_words = 0

            openie_dict = {
                'docs': all_openie_info,
                'avg_ent_chars': avg_ent_chars,
                'avg_ent_words': avg_ent_words
            }

            with open(self.openie_results_path, 'w') as f:
                json.dump(openie_dict, f)
            logger.info(f"OpenAI结果已保存到{self.openie_results_path}")

    def augment_graph(self):
        """
        提供实用函数以通过添加新节点和边来增强图。
        它确保图结构被扩展以包含其他组件，
        并记录完成状态以及打印更新的图信息。
        """

        self.add_new_nodes()
        self.add_new_edges()

        logger.info(f"图构建完成!")
        print(self.get_graph_info())

    def add_new_nodes(self):
        """
        根据其属性，从实体和段落嵌入存储向图中添加新节点。

        此方法通过比较图中现有节点
        和从实体嵌入存储和段落
        嵌入存储中检索的节点来识别和添加新节点。该方法检查属性并确保不添加重复项。
        新节点经过准备并批量添加以优化图更新。
        """

        existing_nodes = {v["name"]: v for v in self.graph.vs if "name" in v.attributes()}

        entity_to_row = self.entity_embedding_store.get_all_id_to_rows()
        passage_to_row = self.chunk_embedding_store.get_all_id_to_rows()

        node_to_rows = entity_to_row
        node_to_rows.update(passage_to_row)

        new_nodes = {}
        for node_id, node in node_to_rows.items():
            node['name'] = node_id
            if node_id not in existing_nodes:
                for k, v in node.items():
                    if k not in new_nodes:
                        new_nodes[k] = []
                    new_nodes[k].append(v)

        if len(new_nodes) > 0:
            self.graph.add_vertices(n=len(next(iter(new_nodes.values()))), attributes=new_nodes)

    def add_new_edges(self):
        """
        从`node_to_node_stats`处理边以将它们添加到图对象中，同时
        管理邻接列表、验证边并记录无效边情况。
        """

        graph_adj_list = defaultdict(dict)
        graph_inverse_adj_list = defaultdict(dict)
        edge_source_node_keys = []
        edge_target_node_keys = []
        edge_metadata = []
        for edge, weight in self.node_to_node_stats.items():
            if edge[0] == edge[1]: continue
            graph_adj_list[edge[0]][edge[1]] = weight
            graph_inverse_adj_list[edge[1]][edge[0]] = weight

            edge_source_node_keys.append(edge[0])
            edge_target_node_keys.append(edge[1])
            edge_metadata.append({
                "weight": weight
            })

        valid_edges, valid_weights = [], {"weight": []}
        current_node_ids = set(self.graph.vs["name"])
        for source_node_id, target_node_id, edge_d in zip(edge_source_node_keys, edge_target_node_keys, edge_metadata):
            if source_node_id in current_node_ids and target_node_id in current_node_ids:
                valid_edges.append((source_node_id, target_node_id))
                weight = edge_d.get("weight", 1.0)
                valid_weights["weight"].append(weight)
            else:
                logger.warning(f"边{source_node_id} -> {target_node_id}无效。")
        self.graph.add_edges(
            valid_edges,
            attributes=valid_weights
        )

    def save_igraph(self):
        logger.info(
            f"写入图，包含{len(self.graph.vs())}个节点，{len(self.graph.es())}条边"
        )
        self.graph.write_pickle(self._graph_pickle_filename)
        logger.info(f"保存图完成!")

    def get_graph_info(self) -> Dict:
        """
        获取有关图的详细信息，例如节点数、
        三元组及其分类。

        此方法基于存储和节点到节点的关系计算有关图的各种统计信息，
        包括短语和段落节点、总节点、提取的三元组、涉及段落节点的三元组、
        同义词三元组和总三元组的计数。

        Returns:
            Dict
                包含以下键及其相应值的字典：
                - num_phrase_nodes: 唯一短语节点数。
                - num_passage_nodes: 唯一段落节点数。
                - num_total_nodes: 总节点数（短语和段落节点之和）。
                - num_extracted_triples: 唯一提取的三元组数。
                - num_triples_with_passage_node: 涉及至少一个
                  段落节点的三元组数。
                - num_synonymy_triples: 同义词三元组数（与提取的
                  三元组和那些带有段落节点的三元组不同）。
                - num_total_triples: 总三元组数。
        """
        graph_info = {}

        # 获取短语节点数
        phrase_nodes_keys = self.entity_embedding_store.get_all_ids()
        graph_info["num_phrase_nodes"] = len(set(phrase_nodes_keys))

        # 获取段落节点数
        passage_nodes_keys = self.chunk_embedding_store.get_all_ids()
        graph_info["num_passage_nodes"] = len(set(passage_nodes_keys))

        # 获取总节点数
        graph_info["num_total_nodes"] = graph_info["num_phrase_nodes"] + graph_info["num_passage_nodes"]

        # 获取提取的三元组数
        graph_info["num_extracted_triples"] = len(self.fact_embedding_store.get_all_ids())

        num_triples_with_passage_node = 0
        passage_nodes_set = set(passage_nodes_keys)
        num_triples_with_passage_node = sum(
            1 for node_pair in self.node_to_node_stats
            if node_pair[0] in passage_nodes_set or node_pair[1] in passage_nodes_set
        )
        graph_info['num_triples_with_passage_node'] = num_triples_with_passage_node

        graph_info['num_synonymy_triples'] = len(self.node_to_node_stats) - graph_info[
            "num_extracted_triples"] - num_triples_with_passage_node

        # 获取总三元组数
        graph_info["num_total_triples"] = len(self.node_to_node_stats)

        return graph_info

    def prepare_retrieval_objects(self):
        """
        准备快速检索过程所需的各种内存对象和属性，例如嵌入数据和图关系，确保一致性
        并与底层图结构对齐。
        """

        logger.info("准备快速检索。")

        logger.info("加载键。")
        self.query_to_embedding: Dict = {'triple': {}, 'passage': {}}

        self.entity_node_keys: List = list(self.entity_embedding_store.get_all_ids()) # 短语节点键列表
        self.passage_node_keys: List = list(self.chunk_embedding_store.get_all_ids()) # 段落节点键列表
        self.fact_node_keys: List = list(self.fact_embedding_store.get_all_ids())

        # 检查图是否具有预期的节点数
        expected_node_count = len(self.entity_node_keys) + len(self.passage_node_keys)
        actual_node_count = self.graph.vcount()

        if expected_node_count != actual_node_count:
            logger.warning(f"图节点计数不匹配: 预期{expected_node_count}，实际{actual_node_count}")
            # 如果图为空但我们有节点，我们需要添加它们
            if actual_node_count == 0 and expected_node_count > 0:
                logger.info(f"用{expected_node_count}个节点初始化图")
                self.add_new_nodes()
                self.save_igraph()

        # 创建从节点名称到顶点索引的映射
        try:
            igraph_name_to_idx = {node["name"]: idx for idx, node in enumerate(self.graph.vs)} # 从节点键到骨干图中的索引
            self.node_name_to_vertex_idx = igraph_name_to_idx

            # 检查所有实体和段落节点是否在图中
            missing_entity_nodes = [node_key for node_key in self.entity_node_keys if node_key not in igraph_name_to_idx]
            missing_passage_nodes = [node_key for node_key in self.passage_node_keys if node_key not in igraph_name_to_idx]

            if missing_entity_nodes or missing_passage_nodes:
                logger.warning(f"图中缺少节点: {len(missing_entity_nodes)}个实体节点，{len(missing_passage_nodes)}个段落节点")
                # 如果缺少节点，重建图
                self.add_new_nodes()
                self.save_igraph()
                # 更新映射
                igraph_name_to_idx = {node["name"]: idx for idx, node in enumerate(self.graph.vs)}
                self.node_name_to_vertex_idx = igraph_name_to_idx

            self.entity_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.entity_node_keys] # 骨干图节点索引列表
            self.passage_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.passage_node_keys] # 骨干段落节点索引列表
        except Exception as e:
            logger.error(f"创建节点索引映射时出错: {str(e)}")
            # 如果映射失败，使用空列表初始化
            self.node_name_to_vertex_idx = {}
            self.entity_node_idxs = []
            self.passage_node_idxs = []

        logger.info("加载嵌入。")
        self.entity_embeddings = np.array(self.entity_embedding_store.get_embeddings(self.entity_node_keys))
        self.passage_embeddings = np.array(self.chunk_embedding_store.get_embeddings(self.passage_node_keys))

        self.fact_embeddings = np.array(self.fact_embedding_store.get_embeddings(self.fact_node_keys))

        all_openie_info, chunk_keys_to_process = self.load_existing_openie([])

        self.proc_triples_to_docs = {}

        for doc in all_openie_info:
            triples = flatten_facts([doc['extracted_triples']])
            for triple in triples:
                if len(triple) == 3:
                    proc_triple = tuple(text_processing(list(triple)))
                    self.proc_triples_to_docs[str(proc_triple)] = self.proc_triples_to_docs.get(str(proc_triple), set()).union(set([doc['idx']]))

        if self.ent_node_to_chunk_ids is None:
            ner_results_dict, triple_results_dict = reformat_openie_results(all_openie_info)

            # 检查长度是否匹配
            if not (len(self.passage_node_keys) == len(ner_results_dict) == len(triple_results_dict)):
                logger.warning(f"长度不匹配: passage_node_keys={len(self.passage_node_keys)}, ner_results_dict={len(ner_results_dict)}, triple_results_dict={len(triple_results_dict)}")

                # 如果缺少键，为它们创建空条目
                for chunk_id in self.passage_node_keys:
                    if chunk_id not in ner_results_dict:
                        ner_results_dict[chunk_id] = NerRawOutput(
                            chunk_id=chunk_id,
                            response=None,
                            metadata={},
                            unique_entities=[]
                        )
                    if chunk_id not in triple_results_dict:
                        triple_results_dict[chunk_id] = TripleRawOutput(
                            chunk_id=chunk_id,
                            response=None,
                            metadata={},
                            triples=[]
                        )

            # 准备data_store
            chunk_triples = [[text_processing(t) for t in triple_results_dict[chunk_id].triples] for chunk_id in self.passage_node_keys]

            self.node_to_node_stats = {}
            self.ent_node_to_chunk_ids = {}
            self.add_fact_edges(self.passage_node_keys, chunk_triples)

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
            # 获取所有查询嵌入
            logger.info(f"为query_to_fact编码{len(all_query_strings)}个查询。")
            query_embeddings_for_triple = self.embedding_model.batch_encode(all_query_strings,
                                                                            instruction=get_query_instruction('query_to_fact'),
                                                                            norm=True)
            for query, embedding in zip(all_query_strings, query_embeddings_for_triple):
                self.query_to_embedding['triple'][query] = embedding

            logger.info(f"为query_to_passage编码{len(all_query_strings)}个查询。")
            query_embeddings_for_passage = self.embedding_model.batch_encode(all_query_strings,
                                                                             instruction=get_query_instruction('query_to_passage'),
                                                                             norm=True)
            for query, embedding in zip(all_query_strings, query_embeddings_for_passage):
                self.query_to_embedding['passage'][query] = embedding

    def get_fact_scores(self, query: str) -> np.ndarray:
        """
        检索并计算给定查询与预存事实嵌入之间的归一化相似度分数。

        Parameters:
        query : str
            需要计算与事实嵌入相似度分数的输入查询文本。

        Returns:
        numpy.ndarray
            查询与事实嵌入之间的归一化相似度分数数组。数组的形状由
            事实的数量决定。

        Raises:
        KeyError
            如果在存储的查询嵌入字典中找不到所提供查询的嵌入。
        """
        query_embedding = self.query_to_embedding['triple'].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(query,
                                                                instruction=get_query_instruction('query_to_fact'),
                                                                norm=True)

        # 检查是否有任何事实
        if len(self.fact_embeddings) == 0:
            logger.warning("没有可用于评分的事实。返回空数组。")
            return np.array([])

        try:
            query_fact_scores = np.dot(self.fact_embeddings, query_embedding.T) # shape: (#facts, )
            query_fact_scores = np.squeeze(query_fact_scores) if query_fact_scores.ndim == 2 else query_fact_scores
            query_fact_scores = min_max_normalize(query_fact_scores)
            return query_fact_scores
        except Exception as e:
            logger.error(f"计算事实分数时出错: {str(e)}")
            return np.array([])

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


    def get_top_k_weights(self,
                          link_top_k: int,
                          all_phrase_weights: np.ndarray,
                          linking_score_map: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        该函数过滤all_phrase_weights以仅保留linking_score_map中排名靠前的短语的权重。
        它还过滤链接分数以仅保留排名前`link_top_k`的节点。未选择的短语在短语
        权重中的权重重置为0.0。

        Args:
            link_top_k (int): 要在链接分数映射中保留的排名靠前的节点数。
            all_phrase_weights (np.ndarray): 表示短语权重的数组，按
                短语ID索引。
            linking_score_map (Dict[str, float]): 将短语内容映射到其链接
                分数的字典，按分数降序排序。

        Returns:
            Tuple[np.ndarray, Dict[str, float]]: 包含过滤后的数组
            of all_phrase_weights，其中未选择的权重设置为0.0，以及过滤后的
            linking_score_map仅包含前`link_top_k`个短语。
        """
        # 在linking_score_map中选择排名靠前的节点
        linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:link_top_k])

        # 仅在all_phrase_weights中保留top_k短语
        top_k_phrases = set(linking_score_map.keys())
        top_k_phrases_keys = set(
            [compute_mdhash_id(content=top_k_phrase, prefix="entity-") for top_k_phrase in top_k_phrases])

        for phrase_key in self.node_name_to_vertex_idx:
            if phrase_key not in top_k_phrases_keys:
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)
                if phrase_id is not None:
                    all_phrase_weights[phrase_id] = 0.0

        assert np.count_nonzero(all_phrase_weights) == len(linking_score_map.keys())
        return all_phrase_weights, linking_score_map

    def graph_search_with_fact_entities(self, query: str,
                                        link_top_k: int,
                                        query_fact_scores: np.ndarray,
                                        top_k_facts: List[Tuple],
                                        top_k_fact_indices: List[str],
                                        passage_node_weight: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        基于事实相似度和相关性使用个性化
        PageRank（PPR）和密集检索模型计算文档分数。此函数结合了与相关事实
        识别的信号与段落相似度和基于图的搜索，以增强结果排序。

        Parameters:
            query (str): 需要执行相似度和相关性计算的
                输入查询字符串。
            link_top_k (int): 要从链接分数映射中包含的排名靠前的短语数
                用于下游处理。
            query_fact_scores (np.ndarray): 表示每个提供的事实的事实-查询相似度
                分数的数组。
            top_k_facts (List[Tuple]): 排名靠前的事实列表，其中每个事实表示
                为主语、谓语和宾语的元组。
            top_k_fact_indices (List[str]): 排名靠前的事实在query_fact_scores数组中
                的相应索引或标识符。
            passage_node_weight (float): 默认权重，用于在图中缩放段落分数。

        Returns:
            Tuple[np.ndarray, np.ndarray]: 包含两个数组的元组：
                - 第一个数组对应于基于其分数排序的文档ID。
                - 第二个数组包含与排序后的文档ID关联的PPR分数。
        """

        # 根据先前步骤中选择的事实分配短语权重
        linking_score_map = {}  # 从短语到包含该短语的事实的平均分数
        phrase_scores = {}  # 存储每个短语的所事实分数，无论它们是否存在于知识图中
        phrase_weights = np.zeros(len(self.graph.vs['name']))
        passage_weights = np.zeros(len(self.graph.vs['name']))
        number_of_occurs = np.zeros(len(self.graph.vs['name']))

        phrases_and_ids = set()

        for rank, f in enumerate(top_k_facts):
            subject_phrase = f[0].lower()
            predicate_phrase = f[1].lower()
            object_phrase = f[2].lower()
            fact_score = query_fact_scores[
                top_k_fact_indices[rank]] if query_fact_scores.ndim > 0 else query_fact_scores

            for phrase in [subject_phrase, object_phrase]:
                phrase_key = compute_mdhash_id(
                    content=phrase,
                    prefix="entity-"
                )
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)

                if phrase_id is not None:
                    weighted_fact_score = fact_score

                    if len(self.ent_node_to_chunk_ids.get(phrase_key, set())) > 0:
                        weighted_fact_score /= len(self.ent_node_to_chunk_ids[phrase_key])

                    phrase_weights[phrase_id] += weighted_fact_score
                    number_of_occurs[phrase_id] += 1

                phrases_and_ids.add((phrase, phrase_id))

        phrase_weights /= number_of_occurs

        for phrase, phrase_id in phrases_and_ids:
            if phrase not in phrase_scores:
                phrase_scores[phrase] = []

            phrase_scores[phrase].append(phrase_weights[phrase_id])

        # 计算每个短语的平均事实分数
        for phrase, scores in phrase_scores.items():
            linking_score_map[phrase] = float(np.mean(scores))

        if link_top_k:
            phrase_weights, linking_score_map = self.get_top_k_weights(link_top_k,
                                                                           phrase_weights,
                                                                           linking_score_map)  # 此时，linking_scope_map的长度由link_top_k确定

        # 根据选择的密集检索模型获取段落分数
        dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.dense_passage_retrieval(query)
        normalized_dpr_sorted_scores = min_max_normalize(dpr_sorted_doc_scores)

        for i, dpr_sorted_doc_id in enumerate(dpr_sorted_doc_ids.tolist()):
            passage_node_key = self.passage_node_keys[dpr_sorted_doc_id]
            passage_dpr_score = normalized_dpr_sorted_scores[i]
            passage_node_id = self.node_name_to_vertex_idx[passage_node_key]
            passage_weights[passage_node_id] = passage_dpr_score * passage_node_weight
            passage_node_text = self.chunk_embedding_store.get_row(passage_node_key)["content"]
            linking_score_map[passage_node_text] = passage_dpr_score * passage_node_weight

        # 将短语和段落分数组合为一个数组用于PPR
        node_weights = phrase_weights + passage_weights

        # 在linking_score_map中记录前30个事实
        if len(linking_score_map) > 30:
            linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:30])

        assert sum(node_weights) > 0, f'给定的事实在图中未找到短语: {top_k_facts}'

        # 基于先前分配的段落和短语权重运行PPR算法
        ppr_start = time.time()
        ppr_sorted_doc_ids, ppr_sorted_doc_scores = self.run_ppr(node_weights, damping=self.global_config.damping)
        ppr_end = time.time()

        self.ppr_time += (ppr_end - ppr_start)

        assert len(ppr_sorted_doc_ids) == len(
            self.passage_node_idxs), f"文档概率长度{len(ppr_sorted_doc_ids)} != 语料库长度{len(self.passage_node_idxs)}"

        return ppr_sorted_doc_ids, ppr_sorted_doc_scores


    def rerank_facts(self, query: str, query_fact_scores: np.ndarray) -> Tuple[List[int], List[Tuple], dict]:
        """
        重排序事实。

        Args:
            query (str): 查询字符串。
            query_fact_scores (np.ndarray): 查询事实分数数组。

        Returns:
            top_k_fact_indices: 排名靠前的事实索引列表。
            top_k_facts: 排名靠前的事实列表。
            rerank_log (dict): 重排序日志字典。
                - facts_before_rerank (list): 重排序前的候选事实列表（每个事实是元组数据类型的关系三元组）。
                - facts_after_rerank (list): 重排序后的排名靠前的事实列表。
        """
        # 加载参数
        link_top_k: int = self.global_config.linking_top_k

        # 检查是否有任何事实需要重排序
        if len(query_fact_scores) == 0 or len(self.fact_node_keys) == 0:
            logger.warning("没有可用于重排序的事实。返回空列表。")
            return [], [], {'facts_before_rerank': [], 'facts_after_rerank': []}

        try:
            # 按分数获取前k个事实
            if len(query_fact_scores) <= link_top_k:
                # 如果我们有比请求更少的事实，使用全部
                candidate_fact_indices = np.argsort(query_fact_scores)[::-1].tolist()
            else:
                # 否则获取前k个
                candidate_fact_indices = np.argsort(query_fact_scores)[-link_top_k:][::-1].tolist()

            # 获取实际的事实ID
            real_candidate_fact_ids = [self.fact_node_keys[idx] for idx in candidate_fact_indices]
            fact_row_dict = self.fact_embedding_store.get_rows(real_candidate_fact_ids)
            candidate_facts = [eval(fact_row_dict[id]['content']) for id in real_candidate_fact_ids]

            # 重排序事实
            top_k_fact_indices, top_k_facts, reranker_dict = self.rerank_filter(query,
                                                                                candidate_facts,
                                                                                candidate_fact_indices,
                                                                                len_after_rerank=link_top_k)

            rerank_log = {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}

            return top_k_fact_indices, top_k_facts, rerank_log

        except Exception as e:
            logger.error(f"rerank_facts中出错: {str(e)}")
            return [], [], {'facts_before_rerank': [], 'facts_after_rerank': [], 'error': str(e)}

    def run_ppr(self,
                reset_prob: np.ndarray,
                damping: float =0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        在图上运行个性化PageRank（PPR），并为对应于文档段落的节点计算相关性分数。
        该方法利用阻尼
        因子用于计算期间的传送，并且可以采用重置概率数组来影响计算的起始状态。

        Parameters:
            reset_prob (np.ndarray): 指定每个节点的重置
                概率分布的一维数组。数组的大小必须等于
                图中的节点数。数组中的NaN或负值
                被替换为零。
            damping (float): 指定计算阻尼因子的
                标量。如果未提供或设置为`None`，默认为0.5。

        Returns:
            Tuple[np.ndarray, np.ndarray]: 包含两个numpy数组的元组。第一个
                数组表示基于其相关性分数降序排列的文档段落的节点ID。第二个
                数组包含同一顺序中每个文档段落的相应相关性分数。
        """

        if damping is None: damping = 0.5 # 为了潜在的兼容性
        reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)
        pagerank_scores = self.graph.personalized_pagerank(
            vertices=range(len(self.node_name_to_vertex_idx)),
            damping=damping,
            directed=False,
            weights='weight',
            reset=reset_prob,
            implementation='prpack'
        )

        doc_scores = np.array([pagerank_scores[idx] for idx in self.passage_node_idxs])
        sorted_doc_ids = np.argsort(doc_scores)[::-1]
        sorted_doc_scores = doc_scores[sorted_doc_ids.tolist()]

        return sorted_doc_ids, sorted_doc_scores

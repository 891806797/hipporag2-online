import json
import difflib
from pydantic import BaseModel, Field, TypeAdapter
from openai import OpenAI
from copy import deepcopy
from typing import Union, Optional, List, Dict, Any, Tuple, Literal
import re
import ast
from .prompts.filter_default_prompt import best_dspy_prompt

class Fact(BaseModel):
    """
    事实数据模型。
    """
    fact: list[list[str]] = Field(description="事实列表，每个事实是包含3个字符串的列表: [主语, 谓语, 宾语]")


class DSPyFilter:
    """
    DSPy过滤器类，用于重排序事实。
    """
    def __init__(self, hipporag):
        """
        使用必要的配置和模板初始化对象，用于处理输入和输出消息。

        Parameters:
        hipporag : 提供全局配置和LLM模型所需推理的对象。

        Attributes:
        dspy_file_path : 全局配置中指定的重排序文件路径。
        one_input_template : 用于格式化输入消息的字符串模板，具有特定字段的占位符。
        one_output_template : 用于格式化输出消息的字符串模板，具有特定字段。
        message_template : 使用指定的dspy文件路径生成的模板。
        llm_infer_fn : 使用提供的LLM模型进行推理的函数引用。
        model_name : 全局配置中指定的语言模型名称。
        default_gen_kwargs : 用于存储默认生成关键字参数的字典。
        """
        dspy_file_path = hipporag.global_config.rerank_dspy_file_path
        self.one_input_template = """[[ ## 问题 ## ]]\n{question}\n\n[[ ## 过滤前的事实 ## ]]\n{fact_before_filter}\n\n以字段`[[ ## 过滤后的事实 ## ]]`（必须格式化为有效的Python Fact）开始响应，然后以`[[ ## 完成 ## ]]`标记结束。"""
        self.one_output_template = """[[ ## 过滤后的事实 ## ]]\n{fact_after_filter}\n\n[[ ## 完成 ## ]]"""
        self.message_template = self.make_template(dspy_file_path)
        self.llm_infer_fn = hipporag.llm_model.infer
        self.model_name = hipporag.global_config.llm_name
        self.default_gen_kwargs = {}

    def make_template(self, dspy_file_path):
        """
        根据dspy文件路径创建消息模板。

        Args:
            dspy_file_path: dspy文件的路径。

        Returns:
            消息模板列表。
        """
        if dspy_file_path is not None:
            dspy_saved = json.load(open(dspy_file_path, 'r'))
        else:
            dspy_saved = best_dspy_prompt

        system_prompt = dspy_saved['prog']['system']
        message_template = [
            {"role": "system", "content": system_prompt},
        ]
        demos = dspy_saved["prog"]["demos"]
        for demo in demos:
            message_template.append({"role": "user", "content": self.one_input_template.format(question=demo["question"], fact_before_filter=demo["fact_before_filter"])})
            message_template.append({"role": "assistant", "content": self.one_output_template.format(fact_after_filter=demo["fact_after_filter"])})
        return message_template

    def parse_filter(self, response):
        """
        解析过滤后的响应。

        Args:
            response: LLM响应字符串。

        Returns:
            解析后的事实列表。
        """
        sections = [(None, [])]
        field_header_pattern = re.compile('\\[\\[ ## (\\w+) ## \\]\\]')
        for line in response.splitlines():
            match = field_header_pattern.match(line.strip())
            if match:
                sections.append((match.group(1), []))
            else:
                sections[-1][1].append(line)

        sections = [(k, "\n".join(v).strip()) for k, v in sections]
        parsed = []
        for k, value in sections:
            if k == "fact_after_filter":
                try:
                    try:
                        parsed_value = json.loads(value)
                    except json.JSONDecodeError:
                        try:
                            parsed_value = ast.literal_eval(value)
                        except (ValueError, SyntaxError):
                            parsed_value = value
                    parsed = TypeAdapter(Fact).validate_python(parsed_value).fact
                except Exception as e:
                    print(
                        f"解析字段{k}时出错: {e}.\n\n\t\t在尝试解析值\n```\n{value}\n```"
                    )

        return parsed

    def llm_call(self, question, fact_before_filter):
        """
        调用LLM进行推理。

        Args:
            question: 问题字符串。
            fact_before_filter: 过滤前的事实JSON字符串。

        Returns:
            LLM响应字符串。
        """
        # 创建提示词
        messages = deepcopy(self.message_template)
        messages.append({"role": "user", "content": self.one_input_template.format(question=question, fact_before_filter=fact_before_filter)})
        # 调用OpenAI

        self.default_gen_kwargs['max_completion_tokens'] = 512

        response = self.llm_infer_fn(
            messages=messages,
            model=self.model_name,
            **self.default_gen_kwargs
        )

        if len(response) > 1:
            return response[0]
        return response

    def __call__(self, *args, **kwargs):
        """
        使对象可调用。

        Args:
            *args: 位置参数。
            **kwargs: 关键字参数。

        Returns:
            重排序结果。
        """
        return self.rerank(*args, **kwargs)

    def rerank(self,
               query: str,
               candidate_items: List[Tuple],
               candidate_indices: List[int],
               len_after_rerank: int =None) -> Tuple[List[int], List[Tuple], dict]:
        """
        重排序候选事实。

        Args:
            query (str): 查询字符串。
            candidate_items (List[Tuple]): 候选事实列表。
            candidate_indices (List[int]): 候选索引列表。
            len_after_rerank (int, optional): 重排序后保留的长度。默认为None。

        Returns:
            Tuple[List[int], List[Tuple], dict]: 包含排序后的索引、排序后的项目和元数据的元组。
        """
        fact_before_filter = {"fact": [list(candidate_item) for candidate_item in candidate_items]}
        try:
            response = self.llm_call(query, json.dumps(fact_before_filter))
            generated_facts = self.parse_filter(response)
        except Exception as e:
            print('异常', e)
            generated_facts = []
        result_indices = []
        for generated_fact in generated_facts:
            closest_matched_fact = difflib.get_close_matches(str(generated_fact), [str(i) for i in candidate_items], n=1, cutoff=0.0)[0]
            try:
                result_indices.append(candidate_items.index(eval(closest_matched_fact)))
            except Exception as e:
                print('result_indices异常', e)

        sorted_candidate_indices = [candidate_indices[i] for i in result_indices]
        sorted_candidate_items = [candidate_items[i] for i in result_indices]
        return sorted_candidate_indices[:len_after_rerank], sorted_candidate_items[:len_after_rerank], {'confidence': None}

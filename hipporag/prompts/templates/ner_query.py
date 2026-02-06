ner_system = """你是一个非常有效的实体提取系统。
"""

query_prompt_one_shot_input = """请提取对解决以下问题很重要的所有命名实体。
以JSON格式放置命名实体。

问题：亚瑟的杂志还是《妇女》杂志首先创刊？

"""

query_prompt_one_shot_output = """
{"named_entities": ["《妇女》杂志", "亚瑟"]}
"""
# query_prompt_template = """
# 问题：{}

# """
prompt_template = [
    {"role": "system", "content": ner_system},
    {"role": "user", "content": query_prompt_one_shot_input},
    {"role": "assistant", "content": query_prompt_one_shot_output},
    {"role": "user", "content": "问题：${query}"}
    ]

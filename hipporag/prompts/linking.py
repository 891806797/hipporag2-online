def get_query_instruction(linking_method):
    """
    获取查询指令。

    Args:
        linking_method (str): 链接方法，可以是'ner_to_node'、'query_to_node'、'query_to_fact'、'query_to_sentence'或'query_to_passage'。

    Returns:
        str: 对应的指令字符串。
    """
    instructions = {
        'ner_to_node': '给定一个短语，检索同义或最匹配该短语的短语。',
        'query_to_node': '给定一个问题，检索该问题中提到的相关短语。',
        'query_to_fact': '给定一个问题，检索与该问题匹配的相关三元组事实。',
        'query_to_sentence': '给定一个问题，检索最好回答该问题的相关句子。',
        'query_to_passage': '给定一个问题，检索最好回答该问题的相关文档。',
    }
    default_instruction = '给定一个问题，检索最好回答该问题的相关文档。'
    return instructions.get(linking_method, default_instruction)

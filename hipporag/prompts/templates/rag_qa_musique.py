# from `gold_with_3_distractors_context_cot_qa_codex.txt`
one_shot_rag_qa_docs = (
    """维基百科标题：最后的马
最后的马（西班牙语：El último caballo）是1950年由埃德加·内维尔执导，由费尔南多·戈麦斯主演的西班牙喜剧片。\n"""
    """维基百科标题：南安普顿镇，尚佩恩县，伊利诺伊州
南安普顿镇是位于伊利诺伊州尚佩恩县的一个镇区。根据2010年人口普查，其人口为505人，拥有219个家庭。\n"""
    """维基百科标题：内维尔·A·斯坦顿
内维尔·A·斯坦顿（Neville A. Stanton）是南安普顿大学的一位英国人类因素和工效学教授。斯坦顿是特许工程师（C.Eng），特许心理学家（C.Psychol）和特许工效学家（C.ErgHF）。他撰写并编辑了四十多本书，并在三个同行评审期刊上发表了关于主体应用的文章，包括《自然》。他还帮助组织设计了新的人机交互界面，例如捷豹汽车的巡航控制系统。\n"""
)
one_shot_ircot_demo = (
    f'{one_shot_rag_qa_docs}'
    '\n问题：'
    f"内维尔·A·斯坦顿的雇主是什么？"
    '\n思考：'
    f"内维尔·A·斯坦顿的雇主是南安普顿大学。南安普顿大学成立于1862年，于1952年获得皇家特许状。所以答案是：1862。"
    '\n\n'
)

rag_qa_system = (
    '作为一个高级阅读理解助手，你的任务是仔细分析文本段落和相应问题。'
    '你将通过系统性的方法分解推理过程，展示你如何得出结论。'
    '你的响应将在"Thought: "之后开始，以"Answer: "呈现简洁、确定的回答。'
)

one_shot_rag_qa_input = (
    f"{one_shot_rag_qa_docs}"
    "\n\n问题："
    "内维尔·A·斯坦顿的雇主是什么？"
    "\n思考："
    "内维尔·A·斯坦顿的雇主是南安普顿大学。南安普顿大学成立于1862年，于1952年获得皇家特许状。所以答案是：1862。"
    "\n\n"
)

one_shot_rag_qa_output = "雇主是南安普顿大学。"

prompt_template = [
    {"role": "system", "content": rag_qa_system},
    {"role": "user", "content": one_shot_rag_qa_input},
    {"role": "assistant", "content": one_shot_rag_qa_output},
    {"role": "user", "content": "${prompt_user}"}
]

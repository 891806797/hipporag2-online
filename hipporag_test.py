if __name__ == '__main__':
    from hipporag import HippoRAG
    from hipporag.utils.config_utils import BaseConfig
    config = BaseConfig(
        llm_name="qwen3-max",
        llm_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        llm_api_key="sk-cac0e78bacee4e93ae09441de234c2c1",
        max_new_tokens=4096,
        temperature=0.0,
        embedding_model_name="text-embedding-v4",
        embedding_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        embedding_api_key="sk-cac0e78bacee4e93ae09441de234c2c1",
        graph_type="facts_and_sim_passage_node_unidirectional",
        save_dir="D:/worker/msun/projects/agents/msun_csm_ai/outputs/xianni",
        save_openie=True
    )

    save_dir = "D:/worker/msun/projects/agents/msun_csm_ai/outputs/xianni"

    hipporag = HippoRAG(save_dir=save_dir, global_config=config)

    # 准备文档
    docs = [
        "王林从小聪明，是村子里公认的神童。",
        "铁柱是王林的小名，他身体瘦弱。",
        "恒岳派收下了王林作为弟子，从此开始修行之路。",
    ]

    # 索引文档
    hipporag.index(docs=docs)

    # 查询
    queries = [
        "谁是王林？他的主要人际关系是什么样？",
        "王林和铁柱的关系是什么？",
    ]

    # 问答
    res = hipporag.rag_qa(queries=queries)
    print("=== answers ===")
    answers = res[0]
    whys = res[1]
    tokens = res[2]
    for i in range(len(queries)):
        print(f"query: {queries[i]}")
        print(f"answer: {answers[i].answer}")
        print(f"why: {whys[i]}")
        print(f"tokens: {tokens[i]}")
        print("================================")
import gradio as gr
from nlp_rag_core import NLP_RAG

# 初始化RAG系统
nlp_rag = NLP_RAG()


# 定义交互函数
def baseline_qa(query, history):
    history = history or []
    answer = nlp_rag.baseline_rag_qa(query)
    history.append((query, answer))
    return history, history


def graph_qa(query, history):
    history = history or []
    answer = nlp_rag.graph_rag_qa(query)
    history.append((query, answer))
    return history, history


def build_graph():
    try:
        nlp_rag.build_knowledge_graph()
        nlp_rag.generate_graph_communities()
        return "知识图谱构建完成！可在Neo4j浏览器查看"
    except Exception as e:
        return f"构建失败：{str(e)}"


# 创建Gradio界面
with gr.Blocks(title="NLP课程答疑智能体") as demo:
    gr.Markdown("# 自然语言处理课程答疑系统")
    gr.Markdown("支持基线RAG和GraphRAG两种模式，输入问题即可获取答案")

    with gr.Tabs():
        with gr.Tab("基线RAG"):
            baseline_chat = gr.Chatbot(label="对话历史")
            baseline_input = gr.Textbox(label="输入问题", placeholder="什么是Word2Vec？")
            baseline_clear = gr.ClearButton([baseline_input, baseline_chat])
            baseline_input.submit(
                baseline_qa,
                [baseline_input, baseline_chat],
                [baseline_chat, baseline_chat]
            )

        with gr.Tab("GraphRAG"):
            graph_chat = gr.Chatbot(label="对话历史")
            graph_input = gr.Textbox(label="输入问题", placeholder="文本分类需要哪些模型？")
            graph_clear = gr.ClearButton([graph_input, graph_chat])
            graph_btn = gr.Button("构建知识图谱（首次运行需点击）")
            graph_status = gr.Textbox(label="构建状态", interactive=False)

            graph_input.submit(
                graph_qa,
                [graph_input, graph_chat],
                [graph_chat, graph_chat]
            )
            graph_btn.click(build_graph, outputs=graph_status)

if __name__ == "__main__":
    try:
        demo.launch(server_name="0.0.0.0", server_port=7860)
    finally:
        nlp_rag.close()  # 关闭数据库连接
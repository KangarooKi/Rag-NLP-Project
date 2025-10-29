import os
import json
from typing import List, Tuple, Dict, Any

import torch
from modelscope import snapshot_download
from langchain.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatZhipuAI  # 若用OpenAI替换为ChatOpenAI
from langchain.prompts import PromptTemplate
from neo4j import GraphDatabase, exceptions

from config import *


class NLP_RAG:
    def __init__(self):
        # 创建数据目录
        os.makedirs(DATA_PATH, exist_ok=True)
        # 加载Embedding模型
        self.embeddings = self._load_embedding_model()
        # 加载向量数据库
        self.vector_db = self._load_vector_db()
        # 加载LLM
        self.llm = self._load_llm()
        # 知识图谱连接
        self.neo4j_driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        # 加载图谱社区摘要
        self.graph_communities = self._load_graph_communities()

    def _load_embedding_model(self) -> HuggingFaceEmbeddings:
        """下载并加载Embedding模型"""
        if not os.path.exists(EMBEDDING_MODEL_PATH):
            print(f"首次运行，下载Embedding模型到 {EMBEDDING_MODEL_PATH}...")
            snapshot_download(
                model_id=EMBEDDING_MODEL_ID,
                cache_dir=EMBEDDING_MODEL_PATH,
                # 将通配符转换为正则表达式（.*匹配任意字符）
                ignore_file_pattern=[r".*\.bin", r".*\.onnx"]
            )
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)

    def _load_vector_db(self) -> Chroma:
        """加载或创建向量数据库"""
        if os.path.exists(VECTOR_DB_PATH):
            print(f"加载已存在的向量库：{VECTOR_DB_PATH}")
            return Chroma(
                persist_directory=VECTOR_DB_PATH,
                embedding_function=self.embeddings
            )
        else:
            print("未找到向量库，开始构建...")
            return self._build_vector_db()

    def _build_vector_db(self) -> Chroma:
        """从文本材料构建向量数据库"""
        # 加载所有文本文件（支持PDF和TXT）
        loaders = [
            DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader),
            DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
        ]
        documents = []
        for loader in loaders:
            documents.extend(loader.load())
        print(f"加载文本文件总数：{len(documents)}")

        # 文本分块
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len
        )
        splits = text_splitter.split_documents(documents)
        print(f"文本分块总数：{len(splits)}")

        # 创建向量库
        vdb = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=VECTOR_DB_PATH
        )
        vdb.persist()
        print(f"向量库构建完成，保存路径：{VECTOR_DB_PATH}")
        return vdb

    def _load_llm(self):
        """加载大语言模型（此处以智谱GLM为例，可替换为其他模型）"""
        return ChatZhipuAI(
            model=LLM_MODEL,
            api_key=LLM_API_KEY,
            temperature=0.5
        )

    def baseline_rag_qa(self, query: str) -> str:
        """基线RAG问答（仅用向量库检索）"""
        # 检索相似文本
        retrieval_results = self.vector_db.similarity_search_with_relevance_scores(
            query=query, k=TOP_K
        )
        formatted_results = "\n".join([
            f"相关文本{idx + 1}（相关性：{score:.2f}）：{doc.page_content}"
            for idx, (doc, score) in enumerate(retrieval_results)
        ])

        # 构建提示词
        prompt_template = """
        你是自然语言处理课程的答疑助手，请严格根据以下材料回答问题：
        1. 只使用提供的材料，不编造信息；
        2. 若材料不相关，直接说明“未找到相关知识点”；
        3. 用学生易懂的语言回答，可举例说明。

        用户问题：{query}
        参考材料：{retrieval_results}
        回答：
        """
        prompt = PromptTemplate.from_template(prompt_template).format(
            query=query, retrieval_results=formatted_results
        )

        # 生成回答
        return self.llm.invoke(prompt).content

    def build_knowledge_graph(self):
        """从文本材料构建知识图谱（实体+关系）"""
        # 1. 从向量库获取所有文本块（字符串列表）
        all_texts = self.vector_db.get()["documents"]  # 直接获取字符串列表
        print(f"向量库中总文本块数量：{len(all_texts)}")

        # 2. 筛选适合提取图谱的文本（包含关系词）
        graph_texts = [
            text for text in all_texts
            if any(keyword in text for keyword in [
                "基于", "依赖", "采用", "借鉴", "依托", "参考",
                "用于", "适用于", "应用于", "服务于", "解决", "针对",
                "组成", "包含", "包括", "由...构成", "涵盖", "涉及",
                "提出", "发明", "设计", "开发", "发布", "推出",
                "属于", "归类于", "作为...的一种", "是...的子任务", "隶属于"
            ])
        ]
        print(f"筛选出可提取图谱的文本块：{len(graph_texts)}")
        if len(graph_texts) == 0:
            print("未找到符合条件的文本块，知识图谱构建终止")
            return

        # 3. 实体关系提取提示词：新增“不换行”“无多余空格”要求（关键修改）
        extract_prompt = PromptTemplate.from_template("""
        从以下文本中提取知识图谱的实体和关系，严格遵守格式要求：
        1. 实体格式：类型:名称（类型仅限：基础概念/算法模型/技术工具/学者/机构/优化问题/变量/函数；名称无多余空格，如“循环网络”）
        2. 关系格式：实体1,关系类型,实体2（关系类型仅限：DependsOn/IsA/UsedFor/ProposedBy/Contains；多个关系用逗号拼接，不换行）
        3. 输出结构：仅一行，格式为“实体：实体1,实体2；关系：关系1,关系2”（不换行，不用分号分隔实体/关系内部）
        4. 强制要求：实体1/实体2必须是已提取的实体名称，无关联则输出“实体：无；关系：无”

        示例正确输出：
        实体：算法模型:MCMC方法,基础概念:概率；关系：MCMC方法,UsedFor,概率,概率,IsA,基础概念

        文本：{text}
        提取结果：
        """)

        # 4. 批量提取并写入Neo4j（循环变量直接用字符串text）
        for idx, text in enumerate(graph_texts):
            try:
                # 调用LLM提取（直接传入text字符串）
                extraction = self.llm.invoke(extract_prompt.format(text=text)).content
                # 解析提取结果
                entities, relationships = self._parse_extraction(extraction)
                # 写入Neo4j（仅当有实体或关系时执行，避免空写入）
                if entities or relationships:
                    self._write_to_neo4j(entities, relationships)
                if (idx + 1) % 10 == 0:
                    print(f"\n===== 已处理 {idx + 1}/{len(graph_texts)} 个文本块 =====")
            except Exception as e:
                print(f"处理文本块 {idx} 失败：{str(e)}")

        print("\n知识图谱构建完成！可在Neo4j浏览器查看实体和关系")

    def _parse_extraction(self, extraction: str) -> Tuple[List, List]:
        entities = []
        relationships = []
        print("\n===== LLM提取原始结果 =====")
        print(extraction)

        # 关键预处理：1. 统一换行符为“；” 2. 去除逗号前后空格 3. 去除多余空行
        extraction = extraction.replace("\n", "；").replace("；；", "；").strip()  # 换行→分号，避免漏解析
        extraction = extraction.replace(", ", ",").replace(" ,", ",").replace("  ", " ")  # 清理逗号空格

        # 第一步：提取所有实体（处理“实体：”开头的内容，兼容换行拆分的情况）
        # 先拆分出所有含“实体：”的片段
        entity_fragments = [part for part in extraction.split("；") if "实体：" in part]
        for frag in entity_fragments:
            frag = frag.strip()
            if not frag:
                continue
            # 提取实体内容（如“实体：算法模型:MCMC方法,基础概念:概率”→“算法模型:MCMC方法,基础概念:概率”）
            ent_content = frag.replace("实体：", "").strip()
            if ent_content == "无" or not ent_content:
                continue
            # 拆分单个实体（按逗号分隔）
            ent_list = [e.strip() for e in ent_content.split(",") if e.strip()]
            for ent in ent_list:
                if ":" in ent:
                    ent_type, ent_name = ent.split(":", 1)  # 只按第一个“:”拆分（避免名称含“:”）
                    ent_type = ent_type.strip()
                    ent_name = ent_name.strip()
                    # 放宽实体类型限制：新增LLM实际输出的类型（如“优化问题”“变量”）
                    allowed_types = [
                        "基础概念", "算法模型", "任务类型", "技术工具", "学者/机构",
                        "优化问题", "变量", "函数", "机构", "学者"
                    ]
                    # 允许实体名含空格（如“循环网络”“整流线性单元”），仅过滤特殊符号
                    if (ent_type in allowed_types or ent_type == "实体") and ent_name and "-" not in ent_name:
                        entities.append(f"{ent_type}:{ent_name}")
                        print(f"解析到实体：{ent_type}:{ent_name}")  # 调试日志：确认实体解析

        # 第二步：提取所有关系（处理“关系：”开头的内容，核心修复批量关系拆分）
        rel_fragments = [part for part in extraction.split("；") if "关系：" in part]
        # 支持的关系类型（新增“Contains”，覆盖日志中“超类Contains词”的情况）
        allowed_rels = ["DependsOn", "IsA", "UsedFor", "ProposedBy", "Contains"]

        for frag in rel_fragments:
            frag = frag.strip()
            if not frag:
                continue
            # 提取关系内容（如“关系：MCMC方法,UsedFor,概率,MCMC方法,UsedFor,状态”→“MCMC方法,UsedFor,概率,MCMC方法,UsedFor,状态”）
            rel_content = frag.replace("关系：", "").strip()
            if rel_content == "无" or not rel_content:
                continue
            print(f"待解析关系内容：{rel_content}")  # 调试日志：查看关系原始内容

            # 核心逻辑：拆分批量关系（如“A,UsedFor,B,A,UsedFor,C”→[A,UsedFor,B; A,UsedFor,C]）
            # 思路：按“关系类型”分割，再拼接成完整三元组
            rel_parts = []
            temp_str = rel_content
            for rel_type in allowed_rels:
                if rel_type in temp_str:
                    # 分割出含当前关系类型的片段（如“MCMC方法,UsedFor,概率,MCMC方法,UsedFor,状态”→分割为“MCMC方法,”和“概率,MCMC方法,”和“状态”）
                    split_by_rel = temp_str.split(rel_type)
                    for i in range(1, len(split_by_rel)):
                        # 拼接实体1 + 关系类型 + 实体2（取前3个有效元素，避免多余内容干扰）
                        prev_part = split_by_rel[i - 1].strip().rstrip(",")  # 实体1（如“MCMC方法”）
                        curr_part = split_by_rel[i].strip().lstrip(",")  # 实体2及后续（如“概率,MCMC方法,UsedFor,状态”）
                        if not prev_part or not curr_part:
                            continue
                        # 提取实体2（取curr_part的第一个元素，忽略后续内容）
                        ent2 = curr_part.split(",")[0].strip()
                        if ent2:
                            rel_parts.append(f"{prev_part},{rel_type},{ent2}")
                    # 更新temp_str，处理剩余关系
                    temp_str = ",".join([split_by_rel[0]] + split_by_rel[1:]).replace(rel_type, "")

            # 过滤有效关系（确保是“实体1,关系,实体2”格式）
            for rel in rel_parts:
                rel = rel.strip()
                if len(rel.split(",")) != 3:
                    print(f"跳过无效关系格式：{rel}")
                    continue
                ent1, rel_type, ent2 = rel.split(",")
                # 校验：关系类型在允许列表，实体1/2非空
                if rel_type in allowed_rels and ent1 and ent2:
                    relationships.append(rel)
                    print(f"解析到关系：{ent1}-[{rel_type}]->{ent2}")  # 调试日志：确认关系解析

        print(f"解析结果：实体{len(entities)}个，关系{len(relationships)}个")
        return entities, relationships

    def _write_to_neo4j(self, entities: List[str], relationships: List[str]):
        with self.neo4j_driver.session() as session:
            entity_names = []  # 记录已成功创建的实体名称
            print("\n===== 开始写入Neo4j =====")

            # 1. 创建实体：捕获单个实体创建错误，避免整体阻塞
            for ent in entities:
                if ":" in ent:
                    ent_type, ent_name = ent.split(":", 1)
                    ent_type = ent_type.strip()
                    ent_name = ent_name.strip()
                    try:
                        # Neo4j标签不允许含空格，若类型有空格（如“学者/机构”），替换为下划线
                        ent_type_safe = ent_type.replace("/", "_").replace(" ", "_")
                        session.run(
                            f"MERGE (e:{ent_type_safe} {{name: $name}}) RETURN e",
                            name=ent_name
                        )
                        entity_names.append(ent_name)
                        print(f"写入实体：{ent_type_safe}:{ent_name}")
                    except Exception as e:
                        print(f"实体写入失败 {ent}：{str(e)}")  # 捕获错误，不影响其他实体

            # 2. 写入关系：校验实体存在性，捕获写入错误
            for rel in relationships:
                if len(rel.split(",")) != 3:
                    print(f"跳过无效关系：{rel}（格式错误）")
                    continue
                ent1, rel_type, ent2 = rel.split(",")
                ent1 = ent1.strip()
                ent2 = ent2.strip()
                rel_type_safe = rel_type.replace("/", "_")  # 关系类型安全处理

                # 校验实体是否已创建
                if ent1 not in entity_names:
                    print(f"关系跳过：实体'{ent1}'未创建（关系：{rel}）")
                    continue
                if ent2 not in entity_names:
                    print(f"关系跳过：实体'{ent2}'未创建（关系：{rel}）")
                    continue

                # 写入关系
                try:
                    session.run(
                        f"""
                        MATCH (a {{name: $ent1}}), (b {{name: $ent2}})
                        MERGE (a)-[r:{rel_type_safe}]->(b)
                        RETURN r
                        """,
                        ent1=ent1, ent2=ent2
                    )
                    print(f"写入关系：{ent1}-[{rel_type_safe}]->{ent2}")
                except Exception as e:
                    print(f"关系写入失败 {rel}：{str(e)}")

            print(
                f"===== Neo4j写入完成：实体{len(entity_names)}个，关系{len([r for r in relationships if len(r.split(',')) == 3])}个 =====")

    def generate_graph_communities(self):
        """生成知识图谱社区摘要"""
        communities = []
        with self.neo4j_driver.session() as session:
            # 获取所有任务类型作为社区核心
            tasks = session.run("MATCH (t:任务类型) RETURN t.name").data()
            for task in tasks:
                task_name = task["t.name"]
                # 查询相关实体
                related_ents = session.run("""
                    MATCH (e)-[r:UsedFor|DependsOn]->(t:任务类型 {name: $task})
                    RETURN e.name AS name, labels(e)[0] AS type
                """, task=task_name).data()
                if related_ents:
                    # 生成摘要
                    ent_desc = "\n".join([f"{e['type']}: {e['name']}" for e in related_ents])
                    summary = self.llm.invoke(f"""
                    总结"{task_name}"相关的NLP知识点（50字内），包括：
                    1. 核心任务：{task_name}
                    2. 关键实体：{ent_desc}
                    """).content
                    communities.append({
                        "name": f"{task_name}社区",
                        "summary": summary,
                        "entities": related_ents
                    })
        # 保存社区摘要
        with open(GRAPH_COMMUNITIES_PATH, "w", encoding="utf-8") as f:
            json.dump(communities, f, ensure_ascii=False, indent=2)
        print(f"社区摘要已保存到 {GRAPH_COMMUNITIES_PATH}")
        return communities

    def _load_graph_communities(self) -> List[Dict]:
        """加载图谱社区摘要"""
        if os.path.exists(GRAPH_COMMUNITIES_PATH):
            with open(GRAPH_COMMUNITIES_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def graph_rag_qa(self, query: str) -> str:
        """GraphRAG问答（融合向量库和知识图谱）"""
        # 获取图谱信息
        graph_info = self._get_graph_info(query)
        # 获取向量库检索结果
        rag_results = self._get_rag_results(query)
        # 融合生成回答
        return self._generate_graph_answer(query, graph_info, rag_results)

    def _get_graph_info(self, query: str) -> str:
        """根据查询获取知识图谱信息"""
        local_keywords = ["关系", "基于", "组成", "提出", "依赖"]
        global_keywords = ["哪些", "总结", "包括", "涉及"]

        if any(kw in query for kw in local_keywords):
            # 局部关系查询
            return self._get_local_graph_info(query)
        elif any(kw in query for kw in global_keywords):
            # 全局社区查询
            return self._get_global_graph_info(query)
        return "未匹配到相关知识图谱信息"

    def _get_local_graph_info(self, query: str) -> str:
        """获取实体间关系信息"""
        # 提取查询中的实体（简化版）
        entities = ["BERT", "Transformer", "Word2Vec", "CRF", "分词", "注意力机制"]
        query_ent = next((e for e in entities if e in query), None)
        if not query_ent:
            return "未识别到具体实体"

        # 查询关系
        with self.neo4j_driver.session() as session:
            rels = session.run("""
                MATCH (a {name: $ent})-[r]->(b)
                RETURN a.name, type(r), b.name
            """, ent=query_ent).data()
        if rels:
            return "实体关系：\n" + "\n".join([
                f"{r['a.name']} {r['type(r)']} {r['b.name']}" for r in rels
            ])
        return "未找到实体关系"

    def _get_global_graph_info(self, query: str) -> str:
        """获取社区摘要信息"""
        matched = [c for c in self.graph_communities if any(kw in query for kw in c["name"])]
        if matched:
            return f"社区摘要（{matched[0]['name']}）：{matched[0]['summary']}"
        return "未找到相关社区"

    def _get_rag_results(self, query: str) -> str:
        """获取向量库检索结果"""
        results = self.vector_db.similarity_search_with_relevance_scores(query, k=TOP_K)
        return "\n".join([
            f"文本{idx + 1}（相关度：{s:.2f}）：{d.page_content}"
            for idx, (d, s) in enumerate(results)
        ])

    def _generate_graph_answer(self, query: str, graph_info: str, rag_results: str) -> str:
        """融合图谱和RAG结果生成回答"""
        prompt = PromptTemplate.from_template("""
        结合知识图谱和文本材料回答NLP课程问题：
        1. 先利用图谱梳理知识点关联，再用文本补充细节；
        2. 分点说明，语言简洁。

        问题：{query}
        知识图谱：{graph_info}
        文本材料：{rag_results}
        回答：
        """).format(query=query, graph_info=graph_info, rag_results=rag_results)
        return self.llm.invoke(prompt).content

    def close(self):
        """关闭资源连接"""
        self.neo4j_driver.close()
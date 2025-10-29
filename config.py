# 1. 路径配置（Windows用D盘或C盘，避免中文！）
DATA_PATH = "D:/nlp_rag_project/data"  # 双反斜杠转义，或用"D:/nlp_rag_project/data"
VECTOR_DB_PATH = f"{DATA_PATH}/nlp_rag_db"  # 向量库会存在这里
GRAPH_COMMUNITIES_PATH = f"{DATA_PATH}/nlp_graph_communities.json"
EMBEDDING_MODEL_PATH = f"{DATA_PATH}/embedding_model/intfloat/e5-base-v2"  # Embedding模型保存路径

# 2. LLM API Key
LLM_API_KEY = "7757aebc9aac42cdaf2d786264acba5b.YlPXrBX03ea3w0XE"  # 从https://www.bigmodel.cn/获取，形如"sk-xxxxxx"

# 3. Neo4j配置（Windows默认和Linux一样，只需改密码）
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "qweasdzxclbq0105"  # 后面安装Neo4j时自己设的密码

# 4. 其他参数（保持默认）
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50
TOP_K = 5
LLM_MODEL = "glm-4-flashx"
EMBEDDING_MODEL_ID = "intfloat/e5-base-v2"
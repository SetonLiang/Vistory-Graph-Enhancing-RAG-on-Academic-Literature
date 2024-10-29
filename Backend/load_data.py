import os,json,re,csv,time
from typing import List, Tuple
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from neo4j import GraphDatabase, basic_auth
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain.load import dumps, loads
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from yfiles_jupyter_graphs import GraphWidget
from typing import Tuple, List, Optional, Union
from operator import itemgetter
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
# 环境变量设置
os.environ["OPENAI_API_BASE"] = 'https://api.chsdw.top/v1'
os.environ["OPENAI_API_KEY"] = "sk-gO7KhknYxgDCHTzC0aE1A4Df0fC040E78c80D296C9FbA001"
os.environ["NEO4J_URI"] = "neo4j+s://d537b991.databases.neo4j.io"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "IXnft6DFgKXaIRnKdszXZDUkGW38tBTUSnJSE3LwSAc"



graph = Neo4jGraph()
# 连接到Neo4j数据库
driver = GraphDatabase.driver(
    uri = os.environ["NEO4J_URI"],
    auth = (os.environ["NEO4J_USERNAME"],
            os.environ["NEO4J_PASSWORD"])
)
session = driver.session()

default_cypher = "MATCH (s)-[r:!HAS_KEYWORD]->(t) RETURN s,r,t LIMIT 50"
def showGraph(cypher: str = default_cypher):
    # create a neo4j session to run queries
    driver = GraphDatabase.driver(
        uri = os.environ["NEO4J_URI"],
        auth = (os.environ["NEO4J_USERNAME"],
                os.environ["NEO4J_PASSWORD"])
    )
    session = driver.session()
    widget = GraphWidget(graph = session.run(cypher).graph())
    widget.node_label_mapping = 'id'
    #display(widget)
    return widget
# 加载作者数据并创建图谱
def load_author_data(department_name, author_name, csv_file):
  
    # 创建作者和部门节点，以及它们之间的关系
    create_author_query = """
        MERGE (d:Department {name: $department_name,type: "Department"})
        MERGE (f:Author {name: $author_name,type: "Author"})
        MERGE (f)-[:BELONGS_TO]->(d)
        RETURN f, d
    """
    session.run(create_author_query, department_name=department_name, author_name=author_name)

    # 读取CSV文件并创建论文节点及其关系
    with open(csv_file, mode='r', encoding='utf-8') as file:
        first_line = file.readline()
        #去除BOM格式
        if first_line.startswith('\ufeff'):
            first_line = first_line.lstrip('\ufeff')

        reader = csv.DictReader(file)
        reader.fieldnames = first_line.strip().split(',')
        for line in reader:
            create_paper_query = """
                MERGE (z:Papers {id: $id, type: "Paper"})
                ON CREATE SET
                     z.year = $year,
                    z.source = $source,
                    z.name = $name,
                    z.authors = $authors,
                    z.abstract = $abstract,
                    z.venue = $venue,
                    z.keywords = $keywords,
                    z.citation = $citation
                MERGE (f:Author {name: $author_name,type: "Author"})
                ON CREATE SET f.name = $author_name
                MERGE (z)-[:OWNED_BY]->(f)
                MERGE (y:Year {name: z.year, type: "Year"})
                MERGE (z)-[:PUBLISHED_IN]->(y)
                MERGE (v:Venue {name: z.source, type: "Venue"})
                MERGE (z)-[:PRESENTED_AT]->(v)
            """
            session.run(create_paper_query, id=line['Id'], year=line['Year'], source=line['Sources'], 
                        name=line['Name'], authors=line['Authors'], abstract=line['Abstract'], 
                        venue=line['Venues'], keywords=line['Keywords'], citation=line['Citation'], author_name=author_name)
                
                
            # 处理关键词并创建关键词节点及关系
            keywords = line['Keywords'].split(',')
            for keyword in keywords:
                if keyword != 'None':
                    create_keyword_query = """
                        MATCH (z:Papers {id: $id})
                        MERGE (k:Keyword {name: $keyword,type: "Keyword"})
                        ON CREATE SET k.name = $keyword
                        MERGE (z)-[:HAS_KEYWORD]->(k)
                    """
                    session.run(create_keyword_query, keyword=keyword.strip(), id=line['Id'])

            
    # 创建论文之间的关键词关系
    create_relationships_query = """
        MATCH (p1:Papers)-[:HAS_KEYWORD]->(k:Keyword)<-[:HAS_KEYWORD]-(p2:Papers)
        WHERE p1 <> p2
        RETURN p1, k, p2
    """
    session.run(create_relationships_query)

   



from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 初始化模型和tokenizer
model_name = "BAAI/bge-m3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
def embed_text(text):
    """将文本嵌入到向量空间中"""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        # 取文本的平均池化作为嵌入
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy().flatten()
def load_embeddings(file_path):
    """从 JSON 文件加载嵌入数据"""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data["nodes"]
def similarity_search(question, file_path, k=10):
    """执行相似度搜索"""
    # 加载节点嵌入
    nodes = load_embeddings(file_path)
    
    # 计算问题的嵌入
    question_embedding = embed_text(question)

    # 提取节点嵌入
    node_embeddings = np.array([np.array(node["embedding"]) for node in nodes])
    
    # 计算相似度
    similarities = cosine_similarity([question_embedding], node_embeddings)[0]

    # 结合相似度和节点信息
    results = []
    for i, node in enumerate(nodes):
        results.append({
            "name": node.get("name"),
            "abstract": node.get("abstract"),
            "keywords": node.get("keywords")
        })
    
    # 按相似度排序并获取前 k 个结果
    sorted_results = sorted(zip(results, similarities), key=lambda x: x[1], reverse=True)
    top_results = [result for result, _ in sorted_results[:k]]
    
    return top_results




if __name__ == "__main__":
    session.run("MATCH (n) DETACH DELETE n")

    # 加载作者和论文数据
    import_dir = 'D://neo4j-community-5.20.0-windows/neo4j-community-5.20.0/import' #指定作者csv文件路径: 部门文件夹-老师.csv
    for department in os.listdir(import_dir):
        department_path = os.path.join(import_dir, department)
        if os.path.isdir(department_path):
            for csv_file in os.listdir(department_path):
                if csv_file.endswith('.csv'):
                    author_name = os.path.splitext(csv_file)[0]
                    csv_file_path = os.path.join(department_path,csv_file)

                    load_author_data(department, author_name, csv_file_path)
                    print(department, author_name, csv_file_path)
 
    print(1)
    
    embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")
    embedding_model = SentenceTransformer("BAAI/bge-m3")
    model_name = "BAAI/bge-m3"
    embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
    )

    # start_time = time.time()

    # 向量索引创建: https://python.langchain.com/v0.2/docs/integrations/vectorstores/neo4jvector/
    session.run("""
             MATCH (p:Papers)
            REMOVE p.embedding
    """               
    )
    try:
        vector_index = Neo4jVector.from_existing_graph(
            # OpenAIEmbeddings(),
            embedding_model,
            search_type="hybrid",
            node_label="Papers",
            text_node_properties=["abstract", "name", "keywords"],
            embedding_node_property="embedding",
            # database="mysql"
        )
        print(vector_index)
    except Exception as e:
        print(f"Error creating vector index: {e}")



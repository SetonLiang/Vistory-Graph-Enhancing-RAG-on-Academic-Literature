import os,json,re,time
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
from trulens_eval import TruChain, Tru




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



# 获取所有节点和标签的映射关系: 存为final_entity_type_map.json文件
def fetch_all_nodes_and_labels():
    map_query = """
    MATCH (n)
    RETURN DISTINCT labels(n) AS labels, n.name AS name
    """
    
    result = session.run(map_query)
    
    node_map = {}
    for record in result:
        name = record.get("name", "").strip().lower()
        labels = record["labels"]
        if name:
            # 假设每个节点只有一个标签
            if labels:
                node_type = labels[0]
                node_map[name] = node_type
            else:
                node_map[name] = "unknown"
                    
    return node_map

def save_map_to_json(node_map, file_path):
    with open(file_path, 'w') as file:
        json.dump(node_map, file, ensure_ascii=False, indent=4)


# 获取每个节点的信息和embedding：存为final_vector_index.json文件
def save_raw_data(file_path):
    nodes_query = "MATCH (n:Papers) RETURN n"
        
    nodes = session.run(nodes_query)
        
    raw_data = {
        "nodes": [record["n"] for record in nodes],
    }
        
    with open(file_path, 'w') as file:
        json.dump(raw_data, file, indent=4)




if __name__ == "__main__":
    node_map = fetch_all_nodes_and_labels()
    save_map_to_json(node_map, "django_HKUST/app01/datasets/final_entity_type_map.json")

    save_raw_data("django_HKUST/app01/datasets/final_vector_index.json")


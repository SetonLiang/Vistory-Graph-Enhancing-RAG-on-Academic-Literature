import os,json,re,csv
from typing import List, Tuple
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    Runnable
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

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app01.views.entity_query import *
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

   

# 实体识别模型
class Entities(BaseModel):
    names: List[str] = Field(
        ...,
        description="All the Department, Author, keyword, Papers, Year and Venue entities that appear in the text",
    )

def generate_full_text_query(input: str) -> str:
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()

def parse_entities(entities: Union[Entities, dict]) -> List[str]:
    if isinstance(entities, Entities):
        return entities.names
    elif isinstance(entities, dict) and 'names' in entities:
        return entities['names']
    else:
        raise ValueError("Invalid entities format")


examples = [
    HumanMessage("How many authors does CMA have?", name="example_user"),
    AIMessage(
        "",
        name="example_assistant",
        tool_calls=[
            {
                "name": "Entities",
                "args": {
                    "names":["CMA"],
                },
                "id": "1",
            }
        ],
    ),
    # Most tool-calling models expect a ToolMessage(s) to follow an AIMessage with tool calls.
    ToolMessage("", tool_call_id="1"),
    # Some models also expect an AIMessage to follow any ToolMessages,
    # so you may need to add an AIMessage here.
    HumanMessage("What are the key contributions of fan mingming's papers?", name="example_user"),
    AIMessage(
        "",
        name="example_assistant",
        tool_calls=[
            {
                "name": "Entities",
                "args": {
                    "names":["fan mingming"],
                },
                "id": "2",
            }
        ],
    ),
    ToolMessage("", tool_call_id="2"),
    HumanMessage("Which papers contain the keyword 'deep learning'", name="example_user"),
    AIMessage(
            "",
            tool_calls=[
                {
                    "name": "Entities",
                    "args": {
                        "names":["deep learning"],
                    },
                    "id": "3",
                }
            ],
        ),
        ToolMessage("", tool_call_id="3"),
]

# 实体识别提示模板
entity_prompt = ChatPromptTemplate.from_messages([
    (
        "system", 
        " You are extracting Department, Author, keyword, Year, Venue and Papers entities from the text.",        
    ),
    (
        "placeholder",
        "{examples}"
    ),
    (
        "human", 
        "Use the given format to extract information from the following"
        "input: {question}",
    ),
])

# 实体链
entity_chain = entity_prompt | ChatOpenAI(temperature=0,model='gpt-4').with_structured_output(Entities)
# print(entity_chain)
# print(entity_chain.invoke({"question":"How many authors does CMA have?","examples":examples}).names)
# print(structured_retriever("What are the key contributions of Kang Zhang's papers?"))



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
    node_embeddings = np.array([np.array(node["embedding"]) for node in nodes if "embedding" in node])
    
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

# def structured_retriever(question: str) -> dict:
#     result = []
#     # 解析问题中的实体
#     entities = entity_chain.invoke({"question": question, "examples": examples})
#     print(entities, type(entities))
#     try:
#         entity_names = parse_entities(entities)
#     except ValueError as e:
#         return {"error": str(e)}
    
#     # 提取年份实体，如果存在的话
#     year_entity = None
#     for entity in entity_names:
#         if entity.isdigit() and len(entity) == 4:  # 简单判断年份
#             year_entity = entity
#             break

#     for entity in entity_names:
#         if '"' in entity:
#             entity = entity.strip('"').replace(",", "").replace(":", "").replace("?", "").replace("-", "").replace(" ", "").lower()
#         else:
#             entity = entity.strip("'").replace(",", "").replace(":", "").replace("?", "").replace("-", "").replace(" ", "").lower()
#         print(entity)

#         def run_to_query(query, params):
#             response = session.run(query, params)
#             return [record.data() for record in response]
#         # 查询作者相关路径
#         author_response = run_to_query(
#             """
#             MATCH (a:Author)-[:OWNED_BY]-(p:Papers)
#             WHERE apoc.text.clean(a.name) = $name
#             OPTIONAL MATCH (p)-[:HAS_KEYWORD]->(k:Keyword)
#             OPTIONAL MATCH (a)-[:BELONGS_TO]->(d:Department)
#             RETURN a, p, collect(k) as keywords, d
#             """,
#             {"name": entity}
#         )
        
#         # 查询论文标题相关路径
#         paper_response = run_to_query(
#             """
#             MATCH (p:Papers)
#             WHERE apoc.text.clean(p.name) = apoc.text.clean($title)
#             OPTIONAL MATCH (p)-[:HAS_KEYWORD]->(k:Keyword)
#             OPTIONAL MATCH (p)-[:OWNED_BY]->(a:Author)
#             RETURN p, collect(k) as keywords, a
#             """,
#             {"title": entity}
#         )
        
#         # 查询关键词相关路径
#         keyword_response = run_to_query(
#             """
#             MATCH (k:Keyword)
#             WHERE apoc.text.clean(k.name) = apoc.text.clean($keyword)
#             OPTIONAL MATCH (k)<-[:HAS_KEYWORD]-(p:Papers)
#             OPTIONAL MATCH (p)-[:OWNED_BY]->(a:Author)
#             RETURN k, collect(p) as papers, collect(a) as authors
#             """,
#             {"keyword": entity}
#         )
        
#         # 查询作者所属部门
#         department_response = run_to_query(
#             """
#             MATCH (d:Department)
#             WHERE apoc.text.clean(d.name) = apoc.text.clean($name)
#             OPTIONAL MATCH (a:Author)-[:BELONGS_TO]->(d)
#             OPTIONAL MATCH (p:Papers)-[:OWNED_BY]->(a)            
#             RETURN a, d, collect(p) as papers
#             """,
#             {"name": entity}
#         )

#         # 处理查询结果
#         if author_response:
#             for record in author_response:
#                 if not year_entity or record['p'].get('year') == year_entity:
#                     result.append({
#                         "type": "author",
#                         "author": record['a'].get('name'),
#                         "paper": record['p'].get('name'),
#                         "year": record['p'].get('year'),
#                         "source": record['p'].get('source'),
#                         "keywords": [k.get('name') for k in record['keywords']],
#                         "department": record['d'].get('name') if record.get('d') else None
#                     })

#         if paper_response:
#             for record in paper_response:
#                 if not year_entity or record['p'].get('year') == year_entity:
#                     result.append({
#                         "type": "paper",
#                         "paper": record['p'].get('name'),
#                         "year": record['p'].get('year'),
#                         "source": record['p'].get('source'),
#                         "author": record['a'].get('name'),
#                         "keywords": [k.get('name') for k in record['keywords']]
#                     })

#         if keyword_response:
#             for record in keyword_response:
#                 filtered_papers = [
#                     {
#                         "name": p.get('name'),
#                         "year": p.get('year'),
#                         "source": p.get('source')
#                     } for p in record['papers'] if not year_entity or p.get('year') == year_entity
#                 ]
#                 if filtered_papers:
#                     result.append({
#                         "type": "keyword",
#                         "keyword": record['k'].get('name'),
#                         "papers": filtered_papers,
#                         "authors": [a.get('name') for a in record['authors']]
#                     })

#         if department_response:
#             total_paper_count = 0
#             for record in department_response:
#                 filtered_papers = [
#                     {
#                         "name": p.get('name'),
#                         "year": p.get('year'),
#                         "source": p.get('source')
#                     } for p in record['papers'] if not year_entity or p.get('year') == year_entity
#                 ]
#                 paper_count = len(filtered_papers)  # 统计每个作者的paper数量
#                 total_paper_count += paper_count
#                 if filtered_papers:
#                     result.append({
#                         "type": "department",
#                         # "author": record['a'].get('name'),
#                         "papers": filtered_papers,
#                         "department": record['d'].get('name'),
#                         "paper_count": paper_count
#                     })
#             result.append({"total_paper_count": total_paper_count})
#     print(result)
#     print(len(result))
#     return {"results": result, "length": len(result), "is_year": year_entity}
def infer_entity_type(entity_name):
    with open("app01/datasets/final_entity_type_map.json","r",encoding='utf-8') as f:
        ENTITY_TYPE_MAP = json.load(f)
    return ENTITY_TYPE_MAP.get(entity_name.lower(), "unknown")

def structured_retriever(question: str) -> dict:
    result = []

    # 使用实体提取工具解析问题中的实体
    entities = entity_chain.invoke({"question": question, "examples": examples})
    
    # 解析出实体名称和类型
    try:
        entity_names = parse_entities(entities)
    except ValueError as e:
        return {"error": str(e)}
    
    # 推断实体类型
    inferred_types = {name: infer_entity_type(name) for name in entity_names}
    print(inferred_types)
     
    # 如果有多个实体，执行联合查询以识别共同关联
    if len(inferred_types) > 1:
        combined_responses = query_combined_entities(inferred_types)
        if combined_responses:
            formatted_result = format_combined_response(combined_responses, inferred_types)
            result.extend(formatted_result)
    else:
        # 构建查询语句
        for entity, entity_type in inferred_types.items():
            
            clean_entity = clean_entity_name(entity)
            # print(clean_entity)
            # 根据实体类型选择相应的查询逻辑
            if entity_type == "Author":
                author_response = query_author(clean_entity)
                if author_response:
                    result.extend(format_author_response(author_response))
                
            elif entity_type == "Papers":
                paper_response = query_paper(clean_entity)
                if paper_response:
                    result.extend(format_paper_response(paper_response))
                
            elif entity_type == "Keyword":
                keyword_response = query_keyword(clean_entity)
                if keyword_response:
                    print(keyword_response)
                    result.extend(format_keyword_response(keyword_response))
                
            elif entity_type == "Department":
                department_response = query_department(clean_entity)
                if department_response:
                    result.extend(format_department_response(department_response))
                
            elif entity_type == "Year":
                year_response = query_year(clean_entity)
                if year_response:
                    result.extend(format_year_response(year_response))

            elif entity_type == "Venue":
                venue_response = query_venue(clean_entity)
                # print(venue_response)
                if venue_response:
                    result.extend(format_venue_response(venue_response))

    return {"results": result, "length": len(result)}

def clean_entity_name(entity):
    """Clean and normalize the entity name."""
    return entity.strip('"\'').replace(",", "").replace(":", "").replace("?", "").replace("-", "").replace(" ", "").lower()

def query_author(author_name):
    """Query for author-related information."""
    query = """
        MATCH (a:Author)-[:OWNED_BY]-(p:Papers)
        WHERE apoc.text.clean(a.name) = apoc.text.clean($name)
        OPTIONAL MATCH (p)-[:HAS_KEYWORD]->(k:Keyword)
        OPTIONAL MATCH (a)-[:BELONGS_TO]->(d:Department)
        RETURN a, p, collect(k) as keywords, d
        Limit 60
    """
    return session.run(query, {"name": author_name})

# def query_paper(paper_title):
#     """Query for paper-related information."""
#     query = """
#         MATCH (p:Papers)
#         WHERE apoc.text.clean(p.name) = apoc.text.clean($title)
#         OPTIONAL MATCH (p)-[:HAS_KEYWORD]->(k:Keyword)
#         OPTIONAL MATCH (p)-[:OWNED_BY]->(a:Author)
#         RETURN p, collect(k) as keywords, a
#     """
#     return session.run(query, {"title": paper_title})
def query_paper(paper_title):
    """Query for paper-related information."""
    query = """
        MATCH (p:Papers)
        WHERE apoc.text.clean(p.name) = apoc.text.clean($title)
        OPTIONAL MATCH (p)-[:HAS_KEYWORD]->(k:Keyword)
        OPTIONAL MATCH (k)<-[:HAS_KEYWORD]-(related_paper:Papers)
        WITH p, k, related_paper
        ORDER BY related_paper.citation DESC
        WITH p, k, collect(related_paper)[0..5] as top_related_papers
        OPTIONAL MATCH (p)-[:OWNED_BY]->(a:Author)
        RETURN p, collect(k) as keywords, a, top_related_papers
    """
    return session.run(query, {"title": paper_title})

def query_keyword(keyword_name):
    """Query for keyword-related information."""
    query = """
        MATCH (k:Keyword)
        WHERE apoc.text.clean(k.name) = apoc.text.clean($keyword)
        OPTIONAL MATCH (k)<-[:HAS_KEYWORD]-(p:Papers)
        OPTIONAL MATCH (p)-[:OWNED_BY]->(a:Author)
        RETURN k, collect(p) as papers, collect(a) as authors
    """
    return session.run(query, {"keyword": keyword_name})

def query_department(department_name):
    """Query for department-related information."""
    query = """
        MATCH (d:Department)
        WHERE apoc.text.clean(d.name) = apoc.text.clean($name)
        OPTIONAL MATCH (a:Author)-[:BELONGS_TO]->(d)
        OPTIONAL MATCH (p:Papers)-[:OWNED_BY]->(a)
        RETURN a, d, collect(p) as papers
    """
    return session.run(query, {"name": department_name})

def query_year(year):
    """Query for year-related information."""
    query = """
        MATCH (y:Year {name: $year})<-[:PUBLISHED_IN]-(p:Papers)
        OPTIONAL MATCH (p)-[:OWNED_BY]->(a:Author)
        OPTIONAL MATCH (a)-[:BELONGS_TO]->(d:Department)
        RETURN a, y, d, collect(p) as papers
    """
    return session.run(query, {"year": year})

def query_venue(venue):
    """Query for venue-related information."""
    query = """
        MATCH (v:Venue)
        WHERE apoc.text.clean(v.name) = apoc.text.clean($venue)
        OPTIONAL MATCH (v)<-[:PRESENTED_AT]-(p:Papers)
        OPTIONAL MATCH (p)-[:OWNED_BY]->(a:Author)
        OPTIONAL MATCH (a)-[:BELONGS_TO]->(d:Department)
        RETURN a, v, d, collect(p) as papers
    """
    return session.run(query, {"venue": venue})


def query_combined_entities(entities):
    """Query for combined entities to find common links."""
    query_author_year = """
        MATCH (a:Author)<-[:OWNED_BY]-(p:Papers)-[:PUBLISHED_IN]->(y:Year)
        WHERE apoc.text.clean(a.name) = apoc.text.clean($author_name)
        AND y.name in $year
        WITH a, p, y
        ORDER BY toInteger(p.citation) DESC  
        RETURN a, collect(p)[0..20] as papers, y 
    """
    """
    1. citation排序
    WITH a, p, y
    ORDER BY p.citations DESC  // Assuming you have a citation or other metric for ordering
    RETURN a, collect(p)[0..20] as papers, y  // Limit to the top 20 papers per year

    2. RETURN a, collect(p) as papers, y
    """

    query_author_keyword = """
        MATCH (a:Author)<-[:OWNED_BY]-(p:Papers)-[:HAS_KEYWORD]->(k:Keyword)
        WHERE apoc.text.clean(a.name) = apoc.text.clean($author_name)
        AND apoc.text.clean(k.name) = apoc.text.clean($keyword)
        RETURN p as papers
    """
    query_author_venue = """
        MATCH (a:Author)<-[:OWNED_BY]-(p:Papers)-[:PRESENTED_AT]->(v:Venue)
        WHERE apoc.text.clean(a.name) = apoc.text.clean($author_name)
        AND apoc.text.clean(v.name) = apoc.text.clean($venue)
        RETURN p as papers
    """
    

    query_department_year = """
        MATCH (d:Department)<-[:BELONGS_TO]-(a:Author)<-[:OWNED_BY]-(p:Papers)-[:PUBLISHED_IN]->(y:Year)
        WHERE apoc.text.clean(d.name) = apoc.text.clean($department_name)
        AND y.name in $year
        WITH d, y, a, p
        ORDER BY y.name, toInteger(p.citation) DESC 
        WITH d, y, a, collect(p)[0..20] as papers
        RETURN d, a, papers, y  
    """
    query_department_keyword = """
        MATCH (d:Department)<-[:BELONGS_TO]-(a:Author)<-[:OWNED_BY]-(p:Papers)-[:HAS_KEYWORD]->(k:Keyword)
        WHERE apoc.text.clean(d.name) = apoc.text.clean($department_name)
        AND apoc.text.clean(k.name) = apoc.text.clean($keyword)
        RETURN a, p as papers
    """
    query_department_venue = """
        MATCH (d:Department)<-[:BELONGS_TO]-(a:Author)<-[:OWNED_BY]-(p:Papers)-[:PRESENTED_AT]->(v:Venue)
        WHERE apoc.text.clean(d.name) = apoc.text.clean($department_name)
        AND apoc.text.clean(v.name) = apoc.text.clean($venue)
        RETURN a, p as papers
    """


    query_keyword_year = """
        MATCH (k:Keyword)<-[:HAS_KEYWORD]-(p:Papers)-[:PUBLISHED_IN]->(y:Year)
        WHERE apoc.text.clean(k.name) = apoc.text.clean($keyword)
        AND y.name = $year
        OPTIONAL MATCH (a:Author)<-[:OWNED_BY]-(p)
        RETURN a, p as papers
    """
    query_keyword_venue = """
        MATCH (k:Keyword)<-[:HAS_KEYWORD]-(p:Papers)-[:PRESENTED_AT]->(v:Venue)
        WHERE apoc.text.clean(k.name) = apoc.text.clean($keyword)
        AND apoc.text.clean(v.name) = apoc.text.clean($venue)
        OPTIONAL MATCH (a:Author)<-[:OWNED_BY]-(p:Papers)
        RETURN a, p as papers
    """

    query_year_venue = """
        MATCH (y:Year)<-[:PUBLISHED_IN]-(p:Papers)-[:PRESENTED_AT]->(v:Venue)
        WHERE y.name = $year
        AND apoc.text.clean(v.name) = apoc.text.clean($venue)
        OPTIONAL MATCH (a:Author)<-[:OWNED_BY]-(p:Papers)
        RETURN a, p as papers
    """

    authors = get_key(entities, 'Author') #判断是否有多个作者
    departments = get_key(entities, 'Department') #判断是否有多个部门


    # Example for combining author and year entities
    if "Author" in entities.values() and "Year" in entities.values():
        author_name = clean_entity_name(get_key(entities,'Author')[0])
        year = get_key(entities,'Year')
        return session.run(query_author_year, {"author_name": author_name, "year": year})
    
    if "Author" in entities.values() and "Keyword" in entities.values():
        author_name = clean_entity_name(get_key(entities,'Author')[0])
        keyword = clean_entity_name(get_key(entities,'Keyword')[0])
        return session.run(query_author_keyword, {"author_name": author_name, "keyword": keyword})
    
    if "Author" in entities.values() and "Venue" in entities.values():
        author_name = clean_entity_name(get_key(entities,'Author')[0])
        venue = clean_entity_name(get_key(entities,'Venue')[0])
        return session.run(query_author_venue, {"author_name": author_name, "venue": venue})
    
    # Example for combining Department and others
    # department can combine with mutiple year=> trend
    if "Department" in entities.values() and "Year" in entities.values():
        department_name = clean_entity_name(get_key(entities,'Department')[0])
        year = get_key(entities,'Year')
        return session.run(query_department_year, {"department_name": department_name, "year": year})
    if "Department" in entities.values() and "Keyword" in entities.values():
        department_name = clean_entity_name(get_key(entities,'Department')[0])
        keyword = clean_entity_name(get_key(entities,'Keyword')[0])
        return session.run(query_department_keyword, {"department_name": department_name, "keyword": keyword})
    if "Department" in entities.values() and "Venue" in entities.values():
        department_name = clean_entity_name(get_key(entities,'Department')[0])
        venue = clean_entity_name(get_key(entities,'Venue')[0])
        return session.run(query_department_venue, {"department_name": department_name, "venue": venue})

    # Example for combining keyword and others
    if "Keyword" in entities.values() and "Year" in entities.values():
        keyword = clean_entity_name(get_key(entities,'Keyword')[0])
        year = get_key(entities,'Year')[0]
        return session.run(query_keyword_year, {"keyword": keyword, "year": year})
    if "Keyword" in entities.values() and "Venue" in entities.values():
        keyword = clean_entity_name(get_key(entities,'Keyword')[0])
        venue = clean_entity_name(get_key(entities,'Venue')[0])
        return session.run(query_keyword_venue, {"keyword": keyword, "venue": venue})

    # Example for combining year and venue
    if "Year" in entities.values() and "Venue" in entities.values():
        year = get_key(entities,'Year')[0]
        venue = clean_entity_name(get_key(entities,'Venue')[0])
        return session.run(query_year_venue, {"year": year, "venue": venue})
    
    if len(authors) > 1:
       # 动态构建MATCH查询部分，包含所有作者
        match_clause = "MATCH " + ", ".join(
            [f"(a{i}:Author)<-[:OWNED_BY]-(p{i}:Papers)" for i in range(len(authors))]
        )
        
        # 构建WHERE子句，用于过滤每个作者的名字
        where_clause = " AND ".join(
            [f"apoc.text.clean(a{i}.name) = apoc.text.clean($author{i}_name)" for i in range(len(authors))]
        )

        # 确保论文节点相同的条件
        paper_match_clause = " AND ".join(
            [f"p0.name = p{i}.name" for i in range(1, len(authors))]
        )

        # 组合完整的Cypher查询
        query_authors = f"""
            {match_clause}
            WHERE {where_clause} AND {paper_match_clause}
            RETURN collect(p0) AS papers
        """
        # 创建参数字典，存储每个作者的名字
        parameters = {f"author{i}_name": clean_entity_name(author) for i, author in enumerate(authors)}
        
        # 执行查询并返回结果
        return session.run(query_authors, parameters)

    if len(departments) > 1:
        # 动态构建MATCH查询部分，包含所有部门
        match_clause = "MATCH " + ", ".join(
            [f"(d{i}:Department)<-[:BELONGS_TO]-(a{i}:Author)<-[:OWNED_BY]-(p{i}:Papers)" for i in range(len(departments))]
        )

        # 构建WHERE子句，用于过滤每个部门的名字
        where_clause = " AND ".join(
            [f"apoc.text.clean(d{i}.name) = apoc.text.clean($department{i}_name)" for i in range(len(departments))]
        )

        # 确保论文节点相同的条件
        paper_match_clause = " AND ".join(
            [f"p0.name = p{i}.name" for i in range(1, len(departments))]
        )

        # 组合完整的Cypher查询
        query_departments = f"""
            {match_clause}
            WHERE {where_clause} AND {paper_match_clause}
            RETURN collect(p0) AS papers
        """

        # 创建参数字典，存储每个部门的名字
        parameters = {f"department{i}_name": clean_entity_name(department) for i, department in enumerate(departments)}

        # 执行查询并返回结果
        return session.run(query_departments, parameters)

    return None


# 提取unstructured_data的relevant paper name
def extract_paper_names(unstructured_data):
    pattern = re.compile(r'name:\s*([^#\n]+)')
    
    paper_names = []
    for entry in unstructured_data:
        matches = pattern.findall(entry)
        paper_names.extend(match.strip() for match in matches)
    
    return paper_names

def unique_dict_list(dict_list):
    # Step 1: Flatten the nested list of dictionaries
    flattened_list = [d for sublist in dict_list for d in sublist]
    
    # Step 2: Remove duplicates using a set of frozensets
    seen = set()
    unique_list = []
    
    for d in flattened_list:
        # Convert the dictionary to a frozenset for uniqueness check
        dict_frozenset = frozenset(d.items())
        if dict_frozenset not in seen:
            seen.add(dict_frozenset)
            unique_list.append(d)
    
    return unique_list


def make_entity_json(relevant_paper,session):
    results = []
    results2 = []
    for item in relevant_paper:
        cypher_query = """
            MATCH (p:Papers {name: $paper_name})
            OPTIONAL MATCH (p)-[r1]->(author:Author)
            OPTIONAL MATCH (p)-[r2]->(keyword:Keyword)
            OPTIONAL MATCH (keyword)<-[rel:HAS_KEYWORD]-() 
            OPTIONAL MATCH (author)-[r3]->(department:Department)
            WHERE (author)-[:BELONGS_TO]->(department)
            RETURN p, author AS a, keyword AS k, department as d, r1, r2, r3, COUNT(rel) AS keyword_count
        """         

        # "RETURN p, author AS a, keyword AS k, department as d, r1, r2, r3, COUNT(r2) AS keyword_count"
        temp = session.execute_read(lambda tx: tx.run(cypher_query, paper_name=item).data())    
        results.append(temp)

    nodes = []
    links = []
    keywords_set = {}
    keyword_counts = {}
    temp_keyword = None
    for lst in results:
        for item in lst:

            temp_paper = {
                'id': item['p']['id'],
                'name': item['p']['name'],
                'released': item['p']['year'],
                'citation': int(item['p']['citation']),
                'authors': item['p']['authors'],
                'venue': item['p']['venue'],
                'group': 0,
                'count': 0,
            }
            # temp_author = {
            #     'name': item['a']['name'],
            #     'group': 1,
            # }
            # if item.get('k'):
            #     temp_keyword = {
            #         'name': item['k']['name'],
            #         'citation': item['keyword_count'],
            #         'group': 2
            #     }
            if item['k']:
                keyword_name = item['k']['name']
                
                keyword_count = item['keyword_count']
                # 使用小写形式作为键
                keyword_name_lower = keyword_name.lower()

                # 检查是否已经存在于 keywords_set 中
                if keyword_name_lower not in keywords_set:
                    keyword_counts[keyword_name_lower]=set()
                    keyword_counts[keyword_name_lower].add(keyword_count)
                    keywords_set[keyword_name_lower] = {
                        'name': keyword_name_lower,
                        'citation': keyword_count,
                        'group': 2
                    }
                else:
                    # if keyword_name_lower == 'visualization':
                    #     print(keyword_name,keyword_count)
                    #     print(keyword_counts[keyword_name_lower],type(keyword_counts[keyword_name_lower]))
                    keyword_counts[keyword_name_lower].add(keyword_count)
                    keywords_set[keyword_name_lower]['citation']=sum(list(keyword_counts[keyword_name_lower]))
                nodes.append(keywords_set[keyword_name_lower])
                
            temp_department = {
                'name': item['d']['name'],
                'group': 1
            }
            temp_link_r1 = {
                'source': item['r1'][2]['name'],
                'target': item['r1'][0]['name'],
                'relationship': item['r1'][1],
            }
            if item['r2']:
                temp_link_r2 = {
                    'source': item['r2'][0]['name'],
                    'target': item['r2'][2]['name'].lower(),
                    'relationship': item['r2'][1],
                }
                links.append(temp_link_r2)
            temp_link_r3 = {
                'source': item['r3'][0]['name'],
                'target': item['r3'][2]['name'],
                'relationship': item['r3'][1],
            }
           
            # nodes.append(temp_author)
            nodes.append(temp_paper)
            nodes.append(temp_department)
            links.append(temp_link_r1)
            
            links.append(temp_link_r3)
    nodes = list(map(dict, set(frozenset(item.items()) for item in nodes)))
    links = list(map(dict, set(frozenset(item.items()) for item in links)))

    for item in relevant_paper:
        cypher_query2 = """
            MATCH (p:Papers {name: $paper_name})
            OPTIONAL MATCH (p)-[r1]->(a:Author)
            RETURN  a
        """
        temp2 = session.execute_read(lambda tx: tx.run(cypher_query2, paper_name=item).data())
        results2.append(temp2)    

    author_list = [[author['a'] for author in lst] for lst in results2]
    # author_list = [list({author['name']: author for author in lst}.values()) for lst in author_list]
    author_list = unique_dict_list(author_list)

    for item in author_list:
        cypher_count_query = '''MATCH (n:Author {name:'%s'})-[:OWNED_BY]-(p) RETURN count(p)''' % item['name']
        with driver.session(database="neo4j") as session:
            count = session.execute_read(lambda tx: tx.run(cypher_count_query, iata="DEN").data())
        item['count'] = count[0]['count(p)']
        item['group'] = 4

    nodes.extend(author_list)

    entity = {
        'nodes': nodes,
        'links': links
    }
    return entity

model_name = "BAAI/bge-m3"
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
)
file_path = "app01/datasets/final_vector_index.json"

def format_unstructured_data(data):
    """将 unstructured_data 格式化为文本"""
    formatted_data = []
    for item in data:
        # 提取需要的信息
        name = item.get('name', 'None')
        abstract = item.get('abstract', 'None')
        keywords = item.get('keywords', 'None')

        # 格式化每个条目
        formatted_data.append(f"name: {name}\nabstract: {abstract}\nkeywords: {keywords}\n")
    return formatted_data

def retriever(question: str):
    # 向量索引创建: https://python.langchain.com/v0.2/docs/integrations/vectorstores/neo4jvector/
    # session.run("""
    #         MATCH (p:Papers)
    #         REMOVE p.embedding
    # """               
    # )
    # try:
    #     vector_index = Neo4jVector.from_existing_graph(
    #         # OpenAIEmbeddings(),
    #         embedding_model,
    #         search_type="hybrid",
    #         node_label="Papers",
    #         text_node_properties=["abstract", "name", "keywords"],
    #         embedding_node_property="embedding",
    #         # database="mysql"
    #     )
    #     print(vector_index)
    # except Exception as e:
    #     print(f"Error creating vector index: {e}")
    # 创建全文索引
    graph.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")

    print(f"Search query: {question}")
    structured_data = structured_retriever(question)
    # is_year = structured_data.get('is_year')
    structured_results = structured_data.get('results', [])
    print(structured_results)
    # unstructured_data = [el.page_content for el in vector_index.similarity_search(question,k=10)] #返回了前五个最相似论文的abstract keyword和name
    unstructured_data = similarity_search(question, file_path, k=5)
    unstructured_data = format_unstructured_data(unstructured_data)

    if structured_results != []:
        relevant_paper = [record.get('paper') for record in structured_results]
    else:
        relevant_paper = extract_paper_names(unstructured_data)
    if len(relevant_paper)>20:
        relevant_paper = relevant_paper[:20]
    print(relevant_paper)
    paper_entity = make_entity_json(relevant_paper,session)
    if question == "What are the latest research findings in the area of virtual reality?":
        with open('app01/datasets/user_study_test1.json', 'w') as f:
            json.dump(paper_entity, f, indent=4)
    else:
        with open('app01/datasets/test.json', 'w') as f:
            json.dump(paper_entity, f, indent=4)
    final_data = f"""Structured data: 
                    {structured_data}
                    Unstructured data:
                    {"#Document ".join(unstructured_data)}
                """
    print(final_data)
    return final_data


CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:""")

def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer

_search_query = RunnableBranch(
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(run_name="HasChatHistoryCheck"),
        RunnablePassthrough.assign(chat_history=lambda x: _format_chat_history(x["chat_history"]))
        | CONDENSE_QUESTION_PROMPT
        | ChatOpenAI(temperature=0)
        | StrOutputParser(),
    ),
    RunnableLambda(lambda x : x["question"]),
)


def return_response(question):
    model_name = "BAAI/bge-m3"
    embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
    )

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

    # 创建全文索引
    graph.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")
    

    # QA提示模版
    answer_template = """Answer the question based only on the following context:
                        {context}
                        Question: {question}
                        Use natural language and be concise.
                        Answer:"""
    answer_prompt = ChatPromptTemplate.from_template(answer_template)

    # QA链
    chain = (
        RunnableParallel({
            "context": _search_query | retriever,
            "question": RunnablePassthrough(),
        })
        | answer_prompt
        | ChatOpenAI(temperature=0)
        | StrOutputParser()
    )

    # 运行QA链
    ans = chain.invoke({"question": question.lower()})

    #利用chat history
    # ans = chain.invoke(
    #     {
    #         "question": "Please give me the paper names",
    #         "chat_history": [("What are the common papers between zengwei and zhangkang?".lower(), "The common papers between Zengwei and Zhangkang involve topics such as visual arts, design, and software engineering. These include works on generative art, landscape rendering, calligraphy, and visual analysis in various contexts. The papers also touch upon themes like visual complexity, security patterns, and interactive art installations.")],
    #     }
    # )
    return ans

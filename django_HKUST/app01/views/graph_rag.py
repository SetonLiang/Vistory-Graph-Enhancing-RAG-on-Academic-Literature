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
        description="All the Department, Author, keyword, Papers and year entities that appear in the text",
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
        " You are extracting Department, Author, keyword, Papers and year entities from the text.",           
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
entity_chain = entity_prompt | ChatOpenAI(temperature=0,model='gpt-3.5-turbo').with_structured_output(Entities)
# print(entity_chain)
# print(entity_chain.invoke({"question":"How many authors does CMA have?","examples":examples}).names)
# print(structured_retriever("What are the key contributions of Kang Zhang's papers?"))



def structured_retriever(question: str) -> dict:
    result = []
    # 解析问题中的实体
    entities = entity_chain.invoke({"question": question, "examples": examples})
    print(entities, type(entities))
    try:
        entity_names = parse_entities(entities)
    except ValueError as e:
        return {"error": str(e)}
    
    # 提取年份实体，如果存在的话
    year_entity = None
    for entity in entity_names:
        if entity.isdigit() and len(entity) == 4:  # 简单判断年份
            year_entity = entity
            break

    for entity in entity_names:
        if '"' in entity:
            entity = entity.strip('"').replace(",", "").replace(":", "").replace("?", "").replace("-", "").replace(" ", "").lower()
        else:
            entity = entity.strip("'").replace(",", "").replace(":", "").replace("?", "").replace("-", "").replace(" ", "").lower()
        print(entity)

        def run_to_query(query, params):
            response = session.run(query, params)
            return [record.data() for record in response]
        # 查询作者相关路径
        author_response = run_to_query(
            """
            MATCH (a:Author)-[:OWNED_BY]-(p:Papers)
            WHERE apoc.text.clean(a.name) = $name
            OPTIONAL MATCH (p)-[:HAS_KEYWORD]->(k:Keyword)
            OPTIONAL MATCH (a)-[:BELONGS_TO]->(d:Department)
            RETURN a, p, collect(k) as keywords, d
            """,
            {"name": entity}
        )
        
        # 查询论文标题相关路径
        paper_response = run_to_query(
            """
            MATCH (p:Papers)
            WHERE apoc.text.clean(p.name) = apoc.text.clean($title)
            OPTIONAL MATCH (p)-[:HAS_KEYWORD]->(k:Keyword)
            OPTIONAL MATCH (p)-[:OWNED_BY]->(a:Author)
            RETURN p, collect(k) as keywords, a
            """,
            {"title": entity}
        )
        
        # 查询关键词相关路径
        keyword_response = run_to_query(
            """
            MATCH (k:Keyword)
            WHERE apoc.text.clean(k.name) = apoc.text.clean($keyword)
            OPTIONAL MATCH (k)<-[:HAS_KEYWORD]-(p:Papers)
            OPTIONAL MATCH (p)-[:OWNED_BY]->(a:Author)
            RETURN k, collect(p) as papers, collect(a) as authors
            """,
            {"keyword": entity}
        )
        
        # 查询作者所属部门
        department_response = run_to_query(
            """
            MATCH (d:Department)
            WHERE apoc.text.clean(d.name) = apoc.text.clean($name)
            OPTIONAL MATCH (a:Author)-[:BELONGS_TO]->(d)
            OPTIONAL MATCH (p:Papers)-[:OWNED_BY]->(a)            
            RETURN a, d, collect(p) as papers
            """,
            {"name": entity}
        )

        # 处理查询结果
        if author_response:
            for record in author_response:
                if not year_entity or record['p'].get('year') == year_entity:
                    result.append({
                        "type": "author",
                        "author": record['a'].get('name'),
                        "paper": record['p'].get('name'),
                        "year": record['p'].get('year'),
                        "source": record['p'].get('source'),
                        "keywords": [k.get('name') for k in record['keywords']],
                        "department": record['d'].get('name') if record.get('d') else None
                    })

        if paper_response:
            for record in paper_response:
                if not year_entity or record['p'].get('year') == year_entity:
                    result.append({
                        "type": "paper",
                        "paper": record['p'].get('name'),
                        "year": record['p'].get('year'),
                        "source": record['p'].get('source'),
                        "author": record['a'].get('name'),
                        "keywords": [k.get('name') for k in record['keywords']]
                    })

        if keyword_response:
            for record in keyword_response:
                filtered_papers = [
                    {
                        "name": p.get('name'),
                        "year": p.get('year'),
                        "source": p.get('source')
                    } for p in record['papers'] if not year_entity or p.get('year') == year_entity
                ]
                if filtered_papers:
                    result.append({
                        "type": "keyword",
                        "keyword": record['k'].get('name'),
                        "papers": filtered_papers,
                        "authors": [a.get('name') for a in record['authors']]
                    })

        if department_response:
            total_paper_count = 0
            for record in department_response:
                filtered_papers = [
                    {
                        "name": p.get('name'),
                        "year": p.get('year'),
                        "source": p.get('source')
                    } for p in record['papers'] if not year_entity or p.get('year') == year_entity
                ]
                paper_count = len(filtered_papers)  # 统计每个作者的paper数量
                total_paper_count += paper_count
                if filtered_papers:
                    result.append({
                        "type": "department",
                        # "author": record['a'].get('name'),
                        "papers": filtered_papers,
                        "department": record['d'].get('name'),
                        "paper_count": paper_count
                    })
            result.append({"total_paper_count": total_paper_count})
    print(result)
    print(len(result))
    return {"results": result, "length": len(result), "is_year": year_entity}

# 提取unstructured_data的relevant paper name
def extract_paper_names(unstructured_data):
    pattern = re.compile(r'name:\s*([^#\n]+)')
    
    paper_names = []
    for entry in unstructured_data:
        matches = pattern.findall(entry)
        paper_names.extend(match.strip() for match in matches)
    
    return paper_names

def unique_dict_list(dict_list):
    seen = set()
    unique_list = []
    for d in dict_list:
        # 将字典转换为 frozenset
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
            RETURN p, author AS a, keyword AS k, r1, r2
        """         
        
        temp = session.execute_read(lambda tx: tx.run(cypher_query, paper_name=item).data())    
        results.append(temp)
    
    nodes = []
    links = []
    for lst in results:
        for item in lst:
            temp_paper = {
                'id': item['p']['id'],
                'name': item['p']['name'],
                'released': item['p']['year'],
                'citation': int(item['p']['citation']),
                'group': 0,
                'count': 0,
            }
            # temp_author = {
            #     'name': item['a']['name'],
            #     'group': 1,
            # }
            if 'k' in item and item['k'] is not None:
                temp_keyword = {
                    'name': item['k']['name'],
                    'group': 2
                }
                temp_link_r2 = {
                    'source': item['r2'][0]['name'],
                    'target': item['r2'][2]['name'],
                    'relationship': item['r2'][1],
                }
                nodes.append(temp_keyword)
                links.append(temp_link_r2)

            temp_link_r1 = {
                'source': item['r1'][2]['name'],
                'target': item['r1'][0]['name'],
                'relationship': item['r1'][1],
            }
                     
            # nodes.append(temp_author)
            nodes.append(temp_paper)
            links.append(temp_link_r1)
            
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
    author_list = unique_dict_list([item[0] for item in author_list])
    
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

    
def retriever(question: str):
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

    print(f"Search query: {question}")
    structured_data = structured_retriever(question)
    is_year = structured_data.get('is_year')
    structured_results = structured_data.get('results', [])

    unstructured_data = [el.page_content for el in vector_index.similarity_search(question,k=10)] #返回了前五个最相似论文的abstract keyword和name

    if is_year:
        relevant_paper = [record.get('paper') for record in structured_results]
    else:
        relevant_paper = extract_paper_names(unstructured_data)
    if len(relevant_paper)>10:
        relevant_paper = relevant_paper[:10]
    print(relevant_paper)
    paper_entity = make_entity_json(relevant_paper,session)
    with open('app01/datasets/test.json', 'w') as f:
        json.dump(paper_entity, f, indent=4)
    final_data = f"""Structured data: 
                    {structured_data}
                    Unstructured data:
                    {"#Document ".join(unstructured_data)}
                """

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

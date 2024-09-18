from collections import defaultdict

from neo4j import GraphDatabase

# Neo4j connection details
URI = "neo4j+s://d537b991.databases.neo4j.io"
AUTH = ("neo4j", "IXnft6DFgKXaIRnKdszXZDUkGW38tBTUSnJSE3LwSAc")
driver = GraphDatabase.driver(URI, auth=AUTH)

# 根据给定作者统计他每年发布文章的数量
query = '''
    MATCH (a:Author {name: $author_name})<-[:OWNED_BY]-(p:Papers)
    WITH p.year AS year, COUNT(p) AS paper_count
    RETURN year, paper_count
    ORDER BY year
    '''

with driver.session(database="neo4j") as session:
    results = session.execute_read(lambda tx: tx.run(query, author_name='Wei Zeng').data())

# 使用 defaultdict 创建一个字典，其中 key 是年份，value 是论文数量
data_dict = defaultdict(int)

# 填充数据
for entry in results:
    year = int(entry['year'])
    paper_count = entry['paper_count']
    data_dict[year] = paper_count

# 将数据转换为所需的格式
data = []
for year, paper_count in sorted(data_dict.items()):
    data.append({"year": year, "paper_count": paper_count})

print(111)

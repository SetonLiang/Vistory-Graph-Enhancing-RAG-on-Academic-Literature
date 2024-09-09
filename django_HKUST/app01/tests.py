from neo4j import GraphDatabase
from collections import defaultdict
# Neo4j connection details
URI = "neo4j+s://d537b991.databases.neo4j.io"
AUTH = ("neo4j", "IXnft6DFgKXaIRnKdszXZDUkGW38tBTUSnJSE3LwSAc")
driver = GraphDatabase.driver(URI, auth=AUTH)

# Define the Cypher query
specified_keyword = "deep learning"

Qquery = '''MATCH (d:Department)<-[:BELONGS_TO]-(a:Author)<-[:OWNED_BY]-(p:Papers)
RETURN d.name AS department, a.name AS author, COUNT(p) AS papers
ORDER BY department, papers DESC'''

# Execute the query and process results
with driver.session(database="neo4j") as session:
    query_results = session.execute_read(lambda tx: tx.run(Qquery, iata="DEN").data())

# 初始化数据结构
data = {"name": "Authors", "children": []}
department_map = defaultdict(list)

# 组织数据
for result in query_results:
    department = result["department"]
    author = result["author"]
    papers = result["papers"]

    department_map[department].append({"name": author, "value": papers})

# 将组织的数据加入到主数据结构中
for department, authors in department_map.items():
    data["children"].append({
        "name": department,
        "children": authors
    })

# Print the result
print(123)

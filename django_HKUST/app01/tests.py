from collections import defaultdict
from neo4j import GraphDatabase

# Neo4j connection details
URI = "neo4j+s://d537b991.databases.neo4j.io"
AUTH = ("neo4j", "IXnft6DFgKXaIRnKdszXZDUkGW38tBTUSnJSE3LwSAc")
driver = GraphDatabase.driver(URI, auth=AUTH)

# Define the Cypher query
specified_keyword = "deep learning"


query = '''
MATCH (k:Keyword)<-[:HAS_KEYWORD]-(p)
WHERE toLower(k.name) = toLower($keyword)
RETURN p AS paper
'''

# Execute the query and process results
with driver.session(database="neo4j") as session:
    results = session.execute_read(lambda tx: tx.run(query, keyword=specified_keyword).data())

# Extract paper names into a list
paper_names = [result['paper']['name'] for result in results]

print(paper_names)
# Print the result
print(123)
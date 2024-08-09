import json

from neo4j import GraphDatabase

URI = "neo4j+s://d537b991.databases.neo4j.io"
AUTH = ("neo4j", "IXnft6DFgKXaIRnKdszXZDUkGW38tBTUSnJSE3LwSAc")

driver = GraphDatabase.driver(URI, auth=AUTH)

cypher_query = '''MATCH (p:Papers) RETURN p.id AS id'''
with driver.session(database="neo4j") as session:
    result = session.execute_read(lambda tx: tx.run(cypher_query, iata="DEN").data())

All_id = []
for paper in result:
    All_id.append(paper['id'])

with open('datasets/All_id.json', 'w') as f:
    json.dump(All_id, f, indent=4)
print('Debugger')
import json

from neo4j import GraphDatabase

URI = "neo4j+s://d537b991.databases.neo4j.io"
AUTH = ("neo4j", "IXnft6DFgKXaIRnKdszXZDUkGW38tBTUSnJSE3LwSAc")

# URI = "neo4j+s://b4c0b390.databases.neo4j.io"
# AUTH = ("neo4j", "0OJWhLSR9PLwVdUUTq1nVpFZ9AZ4yeTrzBjvoqmxl3o")
driver = GraphDatabase.driver(URI, auth=AUTH)

cypher_query = '''MATCH (n:Papers) RETURN n'''
with driver.session(database="neo4j") as session:
    publications = session.execute_read(lambda tx: tx.run(cypher_query, iata="DEN").data())
publications_list = []
for publication in publications:
    publication_dist = {}
    publication_dist.update({'Name': publication['n']['name']})
    publication_dist.update({'Authors': publication['n']['authors'].split(', ')})
    publication_dist.update({'Sources': publication['n']['source']})
    publication_dist.update({'PublishedYears': publication['n']['year']})
    publication_dist.update({'Doi': ''})
    publication_dist.update({'Abstracts': publication['n']['abstract']})
    publication_dist.update({'IdName': publication['n']['id']})
    publication_dist.update({'Citation': ''})
    publication_dist.update({'Keywords': ''})
    publications_list.append(publication_dist)

json_data = json.dumps(publications_list)
print(json_data)
with open('datasets/Init.json', 'w') as f:
    json.dump(publications_list, f)

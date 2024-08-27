from collections import defaultdict

from neo4j import GraphDatabase

URI = "neo4j+s://d537b991.databases.neo4j.io"
AUTH = ("neo4j", "IXnft6DFgKXaIRnKdszXZDUkGW38tBTUSnJSE3LwSAc")

driver = GraphDatabase.driver(URI, auth=AUTH)

min = 0

result = '''
MATCH (a:Author)<-[:OWNED_BY]-(p:Papers)
    RETURN a.name AS authors, COUNT(p) AS papers
        '''

with driver.session(database="neo4j") as session:
    results = session.execute_read(lambda tx: tx.run(result, iata="DEN").data())



print('debugger')

from neo4j import GraphDatabase

URI = "neo4j+s://d537b991.databases.neo4j.io"
AUTH = ("neo4j", "IXnft6DFgKXaIRnKdszXZDUkGW38tBTUSnJSE3LwSAc")

driver = GraphDatabase.driver(URI, auth=AUTH)

min = 0

result = '''MATCH (d:Department)<-[:BELONGS_TO]-(a:Author)<-[:OWNED_BY]-(p:Papers)
WHERE COALESCE(p.citation, '0') > '0'
WITH d.name AS department, p.year AS year, COUNT(p) AS paper_count
ORDER BY department, year
WITH year, COLLECT(paper_count) AS counts
RETURN COLLECT(counts) AS yearly_counts

        '''

with driver.session(database="neo4j") as session:
    results = session.execute_read(lambda tx: tx.run(result, iata="DEN").data())

print('debugger')

from collections import defaultdict
from neo4j import GraphDatabase

# Neo4j connection details
URI = "neo4j+s://d537b991.databases.neo4j.io"
AUTH = ("neo4j", "IXnft6DFgKXaIRnKdszXZDUkGW38tBTUSnJSE3LwSAc")
driver = GraphDatabase.driver(URI, auth=AUTH)

# Define the Cypher query
query = '''
MATCH (k:Keyword)<-[:HAS_KEYWORD]-(p)
WITH k, COUNT(DISTINCT p) AS path_count
RETURN k.name AS keyword, path_count
ORDER BY path_count DESC
LIMIT 100
'''

# Execute the query and process results
with driver.session(database="neo4j") as session:
    results = session.execute_read(lambda tx: tx.run(query).data())

# Use a dictionary to aggregate path counts for each keyword
keyword_counts = defaultdict(int)

for entry in results:
    keyword = entry['keyword'].strip().lower()  # Normalize keyword
    path_count = entry['path_count']
    keyword_counts[keyword] += path_count

# Convert to list of dictionaries with capitalized keywords and the desired format
processed_results = [
    {"text": keyword.capitalize(), "size": count}
    for keyword, count in keyword_counts.items() if keyword  # Exclude empty keywords
]

# Sort results by path count in descending order and limit to top 100
sorted_results = sorted(processed_results, key=lambda x: x['size'], reverse=True)[:100]

# Assign to 'words'
words = sorted_results

# Print the result
print(words)
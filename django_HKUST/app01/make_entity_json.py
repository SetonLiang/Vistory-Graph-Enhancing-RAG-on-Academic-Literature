import json

from neo4j import GraphDatabase

URI = "neo4j+s://d537b991.databases.neo4j.io"
AUTH = ("neo4j", "IXnft6DFgKXaIRnKdszXZDUkGW38tBTUSnJSE3LwSAc")

# URI = "neo4j+s://b4c0b390.databases.neo4j.io"
# AUTH = ("neo4j", "0OJWhLSR9PLwVdUUTq1nVpFZ9AZ4yeTrzBjvoqmxl3o")
driver = GraphDatabase.driver(URI, auth=AUTH)

a = 'Wei Zeng'
b = "Kang Zhang"

cypher_query = '''
    MATCH (d:Department)<-[r1]-(a:Author)
    MATCH (a)<-[r2]-(p:Papers)
    RETURN d, a, p, r1, r2
    '''
with driver.session(database="neo4j") as session:
    results_temp = session.execute_read(lambda tx: tx.run(cypher_query, iata="DEN").data())

nodes = []
links = []
author_list = []
for item in results_temp:
    # temp_author = {
    #     'name': item['a']['name'],
    #     'group': 4
    # }
    temp_paper = {
        'id': item['p']['id'],
        'name': item['p']['name'],
        'released': item['p']['year'],
        'citation': int(item['p']['citation']),
        'authors': item['p']['authors'],
        'venue': item['p']['source'],
        'group': 0,
        'count': 0,
    }
    temp_department = {
        'name': item['d']['name'],
        'group': 1
    }
    temp_link_r1 = {
        'source': item['r1'][2]['name'],
        'target': item['r1'][0]['name'],
        'relationship': item['r1'][1],
    }
    temp_link_r2 = {
        'source': item['r2'][2]['name'],
        'target': item['r2'][0]['name'],
        'relationship': item['r2'][1],
    }
    nodes.append(temp_department)
    # nodes.append(temp_author)
    nodes.append(temp_paper)
    links.append(temp_link_r1)
    links.append(temp_link_r2)

nodes = list(map(dict, set(frozenset(item.items()) for item in nodes)))
links = list(map(dict, set(frozenset(item.items()) for item in links)))

cypher_query_author = '''
        MATCH (n:Author) RETURN n
        '''
with driver.session(database="neo4j") as session:
    results_temp = session.execute_read(lambda tx: tx.run(cypher_query_author, iata="DEN").data())

author_list = [item['n'] for item in results_temp]
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

with open('datasets/entity.json', 'w') as f:
    json.dump(entity, f)

print('debugger')

import json

from django.shortcuts import HttpResponse, render
from neo4j import GraphDatabase

URI = "neo4j+s://d537b991.databases.neo4j.io"
AUTH = ("neo4j", "IXnft6DFgKXaIRnKdszXZDUkGW38tBTUSnJSE3LwSAc")

# URI = "neo4j+s://b4c0b390.databases.neo4j.io"
# AUTH = ("neo4j", "0OJWhLSR9PLwVdUUTq1nVpFZ9AZ4yeTrzBjvoqmxl3o")
driver = GraphDatabase.driver(URI, auth=AUTH)


# Create your views here.

def knowledge_graph(request):
    return render(request, 'graph.html')


def query_all(request):
    with open('app01/datasets/entity.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    return HttpResponse(json.dumps(data))


def query_base(request):
    cypher_query = '''
        MATCH (n:Author) RETURN n
        '''
    with driver.session(database="neo4j") as session:
        results_temp = session.execute_read(lambda tx: tx.run(cypher_query, iata="DEN").data())

    results = [item['n'] for item in results_temp]

    for item in results:
        cypher_count_query = '''MATCH (n:Author {name:'%s'})-[:OWNED_BY]-(p) RETURN count(p)''' % item['name']
        with driver.session(database="neo4j") as session:
            count = session.execute_read(lambda tx: tx.run(cypher_count_query, iata="DEN").data())
        item['count'] = count[0]['count(p)']
        item['group'] = 4

    return HttpResponse(json.dumps(results))


def query_paper(request):
    data = request.GET.get('name')
    cypher_query = 'MATCH (Author {name: "%s"})<-[:OWNED_BY]-(Papers) RETURN Papers' % data
    count = 0
    with driver.session(database="neo4j") as session1:
        results_temp = session1.execute_read(lambda tx: tx.run(cypher_query, iata="DEN").data())
    results = []
    for item in results_temp:
        count += 1
        temp = {
            'id': item['Papers']['id'],
            'name': item['Papers']['name'],
            'released': item['Papers']['year'],
            'citation': int(item['Papers']['citation']),
            'group': 0,
            'count': 0,
        }
        results.append(temp)
    return HttpResponse(json.dumps({'results': results, 'count': count}))


def query_author():
    cypher_query = '''
                MATCH (n:Author) RETURN n
                '''
    with driver.session(database="neo4j") as session:
        results_temp = session.execute_read(lambda tx: tx.run(cypher_query, iata="DEN").data())
    authors_list = [item['n']['name'] for item in results_temp]
    return authors_list


def query_publication():
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
    return publications_list


def query_year():
    cypher_query = '''MATCH (p:Papers)
    RETURN p.year AS group, COUNT(p) AS value
    ORDER BY group
    '''

    with driver.session(database="neo4j") as session:
        result = session.execute_read(lambda tx: tx.run(cypher_query, iata="DEN").data())

    return result


def query_authors_chart():
    cypher_query = '''MATCH (a:Author)<-[:OWNED_BY]-(p:Papers)
    RETURN a.name AS group, COUNT(p) AS value
    '''

    with driver.session(database="neo4j") as session:
        result = session.execute_read(lambda tx: tx.run(cypher_query, iata="DEN").data())

    return result


def query_departments_chart():
    cypher_query_year = '''
    MATCH (p:Papers)
    WITH DISTINCT p.year AS year
    ORDER BY year
    RETURN COLLECT(year) AS years
    '''

    cypher_query_department = '''MATCH (d:Department)
    WITH d.name AS department
    ORDER BY department
    RETURN COLLECT(department) AS departments
        '''

    # cypher_query = '''MATCH (d:Department)<-[:BELONGS_TO]-(a:Author)<-[:OWNED_BY]-(p:Papers)
    # WITH d.name AS department, p.year AS year, COUNT(p) AS paper_count
    # RETURN department, year, paper_count
    # ORDER BY department, year
    # '''

    result = '''MATCH (d:Department)<-[:BELONGS_TO]-(a:Author)<-[:OWNED_BY]-(p:Papers)
    WITH d.name AS department, p.year AS year, COUNT(p) AS paper_count
    ORDER BY department, year
    WITH year, COLLECT(paper_count) AS counts
    RETURN COLLECT(counts) AS yearly_counts
    '''

    with driver.session(database="neo4j") as session:
        year_list = session.execute_read(lambda tx: tx.run(cypher_query_year, iata="DEN").data())[0]['years']
        department_list = session.execute_read(lambda tx: tx.run(cypher_query_department, iata="DEN").data())[0][
            'departments']
        results = session.execute_read(lambda tx: tx.run(result, iata="DEN").data())[0]['yearly_counts']

    return {'x0Name': year_list, 'x1Name': department_list, 'initData': results}


def query_departments_min_chart(min):
    result = '''MATCH (d:Department)<-[:BELONGS_TO]-(a:Author)<-[:OWNED_BY]-(p:Papers)
        WHERE toInteger(p.citation) > %s
        WITH d.name AS department, p.year AS year, COUNT(p) AS paper_count
        ORDER BY department, year
        WITH year, COLLECT(paper_count) AS counts
        RETURN COLLECT(counts) AS yearly_counts
        ''' % int(min - 1)

    with driver.session(database="neo4j") as session:
        results = session.execute_read(lambda tx: tx.run(result, iata="DEN").data())[0]['yearly_counts']

    return {'newData': results}


def flite_authors(authors):
    if len(authors) == 0:
        return []
    elif len(authors) == 1:
        cypher_query = '''MATCH (p:Papers) where p.authors CONTAINS "%s" RETURN p.id''' % authors[0]
    elif len(authors) == 2:
        cypher_query = '''
        MATCH (p:Papers) where p.authors CONTAINS "{}" AND p.authors CONTAINS "{}" RETURN p.id
        '''.format(authors[0], authors[1])
    elif len(authors) == 3:
        cypher_query = '''
        MATCH (p:Papers) where p.authors CONTAINS "{}" 
        AND p.authors CONTAINS "{}" 
        AND p.authors CONTAINS "{}" 
        RETURN p.id
        '''.format(authors[0], authors[1], authors[2])
    elif len(authors) == 4:
        cypher_query = '''
            MATCH (p:Papers) where p.authors CONTAINS "{}" 
            AND p.authors CONTAINS "{}" 
            AND p.authors CONTAINS "{}" 
            AND p.authors CONTAINS "{}" 
            RETURN p.id
            '''.format(authors[0], authors[1], authors[2], authors[3])
    elif len(authors) == 5:
        cypher_query = '''
            MATCH (p:Papers) where p.authors CONTAINS "{}" 
            AND p.authors CONTAINS "{}" 
            AND p.authors CONTAINS "{}" 
            AND p.authors CONTAINS "{}" 
            AND p.authors CONTAINS "{}" 
            RETURN p.id
            '''.format(authors[0], authors[1], authors[2], authors[3], authors[4])
    elif len(authors) == 6:
        cypher_query = '''
            MATCH (p:Papers) where p.authors CONTAINS "{}" 
            AND p.authors CONTAINS "{}" 
            AND p.authors CONTAINS "{}" 
            AND p.authors CONTAINS "{}" 
            AND p.authors CONTAINS "{}" 
            AND p.authors CONTAINS "{}" 
            RETURN p.id
            '''.format(authors[0], authors[1], authors[2], authors[3], authors[4], authors[5])

    with driver.session(database="neo4j") as session1:
        results = session1.execute_read(lambda tx: tx.run(cypher_query, iata="DEN").data())

    results_list = []
    for result in results:
        results_list.append(result['p.id'])

    return results_list


def flite_years(years):
    if len(years) == 1:
        cypher_query = '''MATCH (p:Papers) where p.year CONTAINS "%s" RETURN p.id''' % years[0]

        with driver.session(database="neo4j") as session:
            results = session.execute_read(lambda tx: tx.run(cypher_query, iata="DEN").data())

        results_list = []
        for result in results:
            results_list.append(result['p.id'])

        return results_list
    else:
        return []


def flite_dept(departments):
    print(departments)
    if len(departments) == 0:
        return 0
    elif len(departments) == 1:
        cypher_query = '''MATCH (d:Department {name: "%s"})<-[:BELONGS_TO]-(a:Author)<-[:OWNED_BY]-(p:Papers)
                        RETURN COLLECT(DISTINCT p.id) AS paper_ids''' % departments[0]
    elif len(departments) == 2:
        cypher_query = '''MATCH (a:Author)-[:BELONGS_TO]->(d1:Department {{name: "{}"}}),
      (a)-[:BELONGS_TO]->(d2:Department {{name: "{}"}}),
      (a)<-[:OWNED_BY]-(p:Papers)
        RETURN COLLECT(DISTINCT p.id) AS paper_ids'''.format(departments[0], departments[1])
        print(cypher_query)
    elif len(departments) == 3:
        cypher_query = '''MATCH (a:Author)-[:BELONGS_TO]->(d1:Department {{name: "{}"}}),
        (a)-[:BELONGS_TO]->(d2:Department {{name: "{}"}}),
        (a)-[:BELONGS_TO]->(d3:Department {{name: "{}"}}),
        (a)<-[:OWNED_BY]-(p:Papers)
          RETURN COLLECT(DISTINCT p.id) AS paper_ids'''.format(departments[0], departments[1], departments[2])
    else:
        return 0

    with driver.session(database="neo4j") as session:
        results = session.execute_read(lambda tx: tx.run(cypher_query, iata="DEN").data())[0]['paper_ids']

    return results

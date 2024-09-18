import json
from collections import defaultdict

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
    RETURN p.year AS year, COUNT(p) AS papers
    ORDER BY year
    '''

    with driver.session(database="neo4j") as session:
        result = session.execute_read(lambda tx: tx.run(cypher_query, iata="DEN").data())

    return result


def update_data_for_donut_dept(node):
    # 这个部门有多少个作者
    cypher_query = f'''
    MATCH (d:Department {{name: '{node}'}})<-[:BELONGS_TO]-(a:Author)<-[:OWNED_BY]-(p:Papers)
    WITH d.name AS department, COUNT(DISTINCT a) AS value
    RETURN 
        CASE 
            WHEN department = 'AI' THEN 'Dept.1'
            WHEN department = 'CMA' THEN 'Dept.2'
            WHEN department = 'DSA' THEN 'Dept.3'
            ELSE department
        END AS department,
        value
    UNION
    MATCH (d:Department)
    WHERE d.name <> '{node}'
    RETURN 
        CASE 
            WHEN d.name = 'AI' THEN 'Dept.1'
            WHEN d.name = 'CMA' THEN 'Dept.2'
            WHEN d.name = 'DSA' THEN 'Dept.3'
            ELSE d.name
        END AS department,
        0 AS value
    '''

    with driver.session(database="neo4j") as session:
        results = session.execute_read(lambda tx: tx.run(cypher_query).data())

    # 格式化输出结果
    newData = [{'department': record['department'], 'value': record['value']} for record in results]

    return newData


def update_data_for_chart_dept(node):
    # 这个部门每一年的变化
    query = '''MATCH (d:Department)<-[:BELONGS_TO]-(a:Author)<-[:OWNED_BY]-(p:Papers)
                   WITH d.name AS department, p.year AS year, COUNT(p) AS paper_count
                   RETURN department, year, paper_count
                   ORDER BY department, year'''

    with driver.session(database="neo4j") as session:
        results = session.execute_read(lambda tx: tx.run(query).data())

    # 使用 defaultdict 创建一个字典，其中 key 是年份，value 是另一个字典用于存储各部门的论文数量
    data_dict = defaultdict(lambda: defaultdict(int))

    # 填充数据
    for entry in results:
        year = int(entry['year'])
        department = entry['department']
        paper_count = entry['paper_count']
        data_dict[year][department] = paper_count

    # 将数据转换为所需的格式，并根据目标部门过滤
    data = []
    for year, departments in sorted(data_dict.items()):
        entry = {"year": year}
        # 只保留目标部门的数据，其他部门的数量为 0
        entry[node] = departments.get(node, 0)
        # 对于目标部门之外的部门，确保其数量为 0
        for dept in departments:
            if dept != node:
                entry[dept] = 0
        # 为每个 entry 加入 paper_count: 10
        entry['paper_count'] = 0
        data.append(entry)

    return data


def update_data_for_treemap_dept(node):
    # 定义所有部门的列表
    all_departments = ["DSA", "CMA", "AI"]  # 假设有这三个部门

    # 修改后的 Cypher 查询，增加了部门过滤条件
    cypher_query = '''
        MATCH (d:Department)<-[:BELONGS_TO]-(a:Author)<-[:OWNED_BY]-(p:Papers)
        WHERE d.name = $departmentName
        RETURN d.name AS department, a.name AS author, COUNT(p) AS papers
        ORDER BY papers DESC
        '''

    # 执行查询，传入指定部门名称
    with driver.session(database="neo4j") as session:
        query_results = session.execute_read(lambda tx: tx.run(cypher_query, departmentName=node).data())

    # 初始化根节点为 "Authors"
    data = {"name": "Authors", "children": []}

    # 生成部门数据结构，每个部门作为一个子节点
    for dept in all_departments:
        # 初始化每个部门的 children 为空列表
        department_data = {"name": dept, "children": []}

        # 如果是传入的 node 部门，填充实际的作者数据
        if dept == node:
            for result in query_results:
                author = result["author"]
                papers = result["papers"]
                department_data["children"].append({"name": author, "value": papers})
        else:
            # 对于其他部门，children 为空或者设为 0
            department_data["children"].append({"name": "No data", "value": 0})

        # 将部门数据添加到主结构中
        data["children"].append(department_data)

    return data


def update_data_for_wordcloud_dept(node):
    # 这个部门的词云
    cypher_query = ''''''

    with driver.session(database="neo4j") as session:
        query_results = session.execute_read(lambda tx: tx.run(cypher_query, iata="DEN").data())


def update_data_for_donut_author(node):
    # 这个部门有多少个作者
    cypher_query = f'''
    MATCH (a:Author {{name: '{node}'}})-[:BELONGS_TO]->(d:Department)
    WITH COLLECT(d.name) AS dept_names

    // 返回所有部门的数量
    MATCH (d:Department)
    RETURN 
        CASE 
            WHEN d.name = 'AI' THEN 'Dept.1'
            WHEN d.name = 'CMA' THEN 'Dept.2'
            WHEN d.name = 'DSA' THEN 'Dept.3'
            ELSE d.name
        END AS department,
        CASE 
            WHEN d.name IN dept_names THEN 1
            ELSE 0
        END AS value
    '''

    with driver.session(database="neo4j") as session:
        results = session.execute_read(lambda tx: tx.run(cypher_query).data())

    # 格式化输出结果
    newData = [{'department': record['department'], 'value': record['value']} for record in results]

    return newData


def update_data_for_chart_author(node):
    # 这个作者每一年的变化
    query = '''
        MATCH (a:Author {name: $author_name})<-[:OWNED_BY]-(p:Papers)
        WITH p.year AS year, COUNT(p) AS paper_count
        RETURN year, paper_count
        ORDER BY year
        '''

    with driver.session(database="neo4j") as session:
        results = session.execute_read(lambda tx: tx.run(query, author_name=node).data())

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
        # 每个年份的数据加上 'AI', 'CMA', 'DSA' 初始为 0
        data.append({
            "year": year,
            "paper_count": paper_count,
            "AI": 0,
            "CMA": 0,
            "DSA": 0
        })

    return data


def update_data_for_treemap_author(author_name):
    # 定义所有部门的列表
    all_departments = ["DSA", "CMA", "AI"]  # 假设有这三个部门

    # 查询作者所在的所有部门
    department_query = '''
        MATCH (a:Author {name: $authorName})-[:BELONGS_TO]->(d:Department)
        RETURN d.name AS department
        '''

    # 执行查询，找到作者所在的所有部门
    with driver.session(database="neo4j") as session:
        dept_results = session.execute_read(lambda tx: tx.run(department_query, authorName=author_name).data())

    if not dept_results:
        # 如果作者没有找到部门，返回空的数据结构
        return {"name": "Authors",
                "children": [{"name": dept, "children": [{"name": "No data", "value": 0}]} for dept in all_departments]}

    # 获取作者所在的所有部门
    author_depts = {result["department"] for result in dept_results}

    # 修改后的 Cypher 查询，增加了部门过滤条件
    cypher_query = '''
        MATCH (d:Department)<-[:BELONGS_TO]-(a:Author)<-[:OWNED_BY]-(p:Papers)
        WHERE d.name IN $departmentNames
        RETURN d.name AS department, a.name AS author, COUNT(p) AS papers
        ORDER BY d.name, papers DESC
        '''

    # 执行查询，传入所有相关部门名称
    with driver.session(database="neo4j") as session:
        query_results = session.execute_read(lambda tx: tx.run(cypher_query, departmentNames=list(author_depts)).data())

    # 初始化根节点为 "Authors"
    data = {"name": "Authors", "children": []}

    # 生成部门数据结构，每个部门作为一个子节点
    for dept in all_departments:
        # 初始化每个部门的 children 为空列表
        department_data = {"name": dept, "children": []}

        # 如果是作者所在的部门，填充实际的作者数据
        if dept in author_depts:
            # 添加该部门的作者数据
            dept_data = [result for result in query_results if result["department"] == dept]
            for result in dept_data:
                author = result["author"]
                papers = result["papers"]
                department_data["children"].append({"name": author, "value": papers})
        else:
            # 对于其他部门，children 为空或者设为 0
            department_data["children"].append({"name": "No data", "value": 0})

        # 将部门数据添加到主结构中
        data["children"].append(department_data)

    return data


def update_data_for_wordcloud_author(node):
    # 这个部门的词云
    cypher_query = ''''''

    with driver.session(database="neo4j") as session:
        query_results = session.execute_read(lambda tx: tx.run(cypher_query, iata="DEN").data())


def query_find_paper_by_keyword(keyword):
    specified_keyword = keyword

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
    return paper_names


def query_authors_chart():
    cypher_query = '''MATCH (d:Department)<-[:BELONGS_TO]-(a:Author)<-[:OWNED_BY]-(p:Papers)
RETURN d.name AS department, a.name AS author, COUNT(p) AS papers
ORDER BY department, papers DESC'''

    # Execute the query and process results
    with driver.session(database="neo4j") as session:
        query_results = session.execute_read(lambda tx: tx.run(cypher_query, iata="DEN").data())

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

    return data


def query_keywords():
    cypher_query = '''
    MATCH (k:Keyword)<-[:HAS_KEYWORD]-(p)
    WITH toLower(k.name) AS keyword, COUNT(DISTINCT p) AS path_count
    RETURN keyword, path_count
    ORDER BY path_count DESC
    '''
    with driver.session(database="neo4j") as session:
        results = session.execute_read(lambda tx: tx.run(cypher_query).data())

    words = [
        {"text": result["keyword"], "size": result["path_count"]}
        for result in results
    ]
    return words


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


def query_paper_department():
    query = '''MATCH (d:Department)<-[:BELONGS_TO]-(a:Author)<-[:OWNED_BY]-(p:Papers)
WITH d.name AS department, COUNT(DISTINCT a) AS value
RETURN 
    CASE 
        WHEN department = 'AI' THEN 'Dept.1'
        WHEN department = 'CMA' THEN 'Dept.2'
        WHEN department = 'DSA' THEN 'Dept.3'
        ELSE department
    END AS department,
    value'''
    with driver.session(database="neo4j") as session:
        results = session.execute_read(lambda tx: tx.run(query).data())

    return results


def query_paper_department_year():
    query = '''MATCH (d:Department)<-[:BELONGS_TO]-(a:Author)<-[:OWNED_BY]-(p:Papers)
            WITH d.name AS department, p.year AS year, COUNT(p) AS paper_count
            RETURN department, year, paper_count
            ORDER BY department, year'''

    with driver.session(database="neo4j") as session:
        results = session.execute_read(lambda tx: tx.run(query).data())

    # 使用 defaultdict 创建一个字典，其中 key 是年份，value 是另一个字典用于存储各部门的论文数量
    data_dict = defaultdict(lambda: defaultdict(int))

    # 填充数据
    for entry in results:
        year = int(entry['year'])
        department = entry['department']
        paper_count = entry['paper_count']
        data_dict[year][department] = paper_count

    # 将数据转换为所需的格式
    data = []
    for year, departments in sorted(data_dict.items()):
        entry = {"year": year}
        entry.update(departments)
        entry["paper_count"] = 0  # 为每条数据添加 paper_count: 10
        data.append(entry)

    return data

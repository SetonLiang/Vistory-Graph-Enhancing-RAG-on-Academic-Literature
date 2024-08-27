from collections import defaultdict

from neo4j import GraphDatabase

URI = "neo4j+s://d537b991.databases.neo4j.io"
AUTH = ("neo4j", "IXnft6DFgKXaIRnKdszXZDUkGW38tBTUSnJSE3LwSAc")

driver = GraphDatabase.driver(URI, auth=AUTH)

min = 0

result = '''
MATCH (k:Keyword)<-[:HAS_KEYWORD]-(p)
WITH k, COUNT(DISTINCT p) AS path_count
RETURN k.name AS keyword, path_count
ORDER BY path_count DESC
LIMIT 100
        '''

with driver.session(database="neo4j") as session:
    results = session.execute_read(lambda tx: tx.run(result, iata="DEN").data())

# 使用字典合并相同关键词
keyword_counts = defaultdict(int)

for entry in results:
    keyword = entry['keyword'].strip().lower()  # 转换为小写并去除前后空白
    path_count = entry['path_count']
    keyword_counts[keyword] += path_count

# 将结果转换为列表并将关键词首字母大写
processed_results = [
    [keyword.capitalize(), count]
    for keyword, count in keyword_counts.items() if keyword  # 过滤掉空字符串
]

# 按路径数量从大到小排序并限制结果为前 100 个
sorted_results = sorted(processed_results, key=lambda x: x[1], reverse=True)[:100]

# 打印结果
word_array = sorted_results
print("wordArray =", word_array)

print('debugger')

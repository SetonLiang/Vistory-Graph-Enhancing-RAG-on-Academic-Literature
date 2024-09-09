
def format_author_response(response):
    formatted_results = []
    for record in response:
        formatted_results.append({
            "type": "author",
            "author": record['a'].get('name'),
            "paper": record['p'].get('name'),
            "year": record['p'].get('year'),
            "venue": record['p'].get('source'),
            "citation": record['p'].get('citation'),
            "keywords": [k.get('name') for k in record['keywords']],
            "department": record['d'].get('name') if record.get('d') else None
        })
    return formatted_results
# def format_paper_response(response):
#     formatted_results = []
#     for record in response:
#         formatted_results.append({
#             "type": "paper",
#             "paper": record['p'].get('name'),
#             "year": record['p'].get('year'),
#             "venue": record['p'].get('source'),
#             "author": record['a'].get('name'),
#             "abstract": record['p'].get('abstract'),
#             "keywords": [k.get('name') for k in record['keywords']]
#         })
#     return formatted_results
def format_paper_response(response):
    formatted_results = []
    for record in response:
        for related_paper in record['top_related_papers']:
            formatted_results.append({
                "type": "paper",
                "paper": related_paper.get('name'),
                "year": related_paper.get('year'),
                "venue": related_paper.get('source'),
                "citation": related_paper.get('citation'),
                "author": record['a'].get('name'),
                "abstract": related_paper.get('abstract'),
                "keywords": [k.get('name') for k in record['keywords']]
            })
    return formatted_results
def format_keyword_response(response):
    formatted_results = []
    for record in response:
        for p in record['papers']:  # 遍历每个paper
            formatted_results.append({
                "type": "keyword",
                "keyword": record['k'].get('name'),
                "paper": p.get('name'),
                "year": p.get('year'),
                "venue": p.get('source'),
                "citation": p.get('citation'),
                "authors": [a.get('name') for a in record['authors']]
            })
    return formatted_results

def format_department_response(response):
    formatted_results = []
    total_paper_count = 0
    author_set = set()  # 用于统计唯一作者数量
    
    for record in response:
        author_name = record['a'].get('name')  # 获取作者名称
        department_name = record['d'].get('name')  # 获取部门名称
        papers = record['papers']  # 获取作者的所有论文
        
        # 统计部门内所有论文的总数
        total_paper_count += len(papers)
        author_set.add(author_name)
        
        # 只返回前8篇论文
        limited_papers = papers[:6]
        
        for paper in limited_papers:
            formatted_results.append({
                "author": author_name,
                "paper": paper.get('name'),
                "year": paper.get('year'),
                "venue": paper.get('source'),
                "citation":paper.get('citation'),
                "department": department_name
            })
    
    # 添加统计信息到结果中
    formatted_results.append({"total_paper_count": total_paper_count})
    formatted_results.append({"author_count": len(author_set)})
    
    return formatted_results



def get_key (dict, value):
    return [k for k, v in dict.items() if v == value]

def format_year_response(response):
    formatted_results = []
    total_paper_count = 0
    author_set = set()  # 用于统计唯一作者数量
    
    for record in response:
        author_name = record['a'].get('name')  # 获取作者名称
        department_name = record['d'].get('name')  
        papers = record['papers']  # 获取作者的所有论文
        
        # 统计每年内所有论文的总数
        total_paper_count += len(papers)
        author_set.add(author_name)
        
        # 只返回前8篇论文
        limited_papers = papers[:2]
        
        for paper in limited_papers:
            formatted_results.append({
                "author": author_name,
                "paper": paper.get('name'),
                "venue": paper.get('source'),
                "citation": paper.get('citation'),
                "keywords": paper.get('keywords'),
                "department": department_name
            })
    
    # 添加统计信息到结果中
    formatted_results.append({"total_paper_count": total_paper_count})
    formatted_results.append({"author_count": len(author_set)})
    
    return formatted_results
def format_venue_response(response):
    formatted_results = []
    total_paper_count = 0
    author_set = set()  # 用于统计唯一作者数量
    
    for record in response:
        author_name = record['a'].get('name')  # 获取作者名称
        department_name = record['d'].get('name')  
        papers = record['papers']  # 获取作者的所有论文
        
        # 统计部门内所有论文的总数
        total_paper_count += len(papers)
        author_set.add(author_name)
        
        # 只返回前8篇论文
        limited_papers = papers[:8]
        
        for paper in limited_papers:
            formatted_results.append({
                "author": author_name,
                "paper": paper.get('name'),
                "year": paper.get('year'),
                "citation": paper.get('citation'),
                "keywords": paper.get('keywords'),
                "department": department_name
            })
    
    # 添加统计信息到结果中
    formatted_results.append({"total_paper_count": total_paper_count})
    formatted_results.append({"author_count": len(author_set)})
    
    return formatted_results
def format_author_year_response(response):
    formatted_results = []

    for record in response:
        author_name = record['a'].get('name')  # 获取作者名称
        year_name = record['y'].get('name')  # 获取年份

        # 遍历每篇论文
        for paper in record['papers']:
            formatted_results.append({
                "author": author_name,
                "paper": paper.get('name'),
                "venue": paper.get('source'),
                "citation": paper.get('citation'),
                "keywords": paper.get('keywords'),  # 假设关键词是一个单一值或字符串
                "year": paper.get("year")
                # "abstract": paper.get('abstract')
            })

    return formatted_results


def format_author_author_response(responses):
    formatted_results = []

    for record in responses:
        # 遍历每篇论文
        for paper in record['papers']:
            formatted_results.append({
                "paper": paper.get('name'),
                "venue": paper.get('source'),
                "citation": paper.get('citation'),
                "keywords": paper.get('keywords')  # 假设关键词是单一值或字符串
            })

    return formatted_results

def format_author_keyword_response(responses):
    formatted_results = []

    for record in responses:
        papers = record['papers']  # 获取论文信息
        
        formatted_results.append({
            "paper": papers.get('name'),
            "venue": papers.get('source'),
            "year": papers.get('year'),
            "citation": papers.get('citation'),
            "keywords": papers.get('keywords')  # 假设关键词是一个单一值或字符串
        })

    return formatted_results

def format_author_venue_response(responses):
    formatted_results = []

    for record in responses:
        papers = record['papers']  # 获取论文信息
        
        formatted_results.append({
            "paper": papers.get('name'),
            "year": papers.get('year'),
            "citation": papers.get('citation'),
            "keywords": papers.get('keywords')  # 假设关键词是一个单一值或字符串
            
        })

    return formatted_results

def format_department_year_response(responses):
    formatted_results = []
    author_set = set() 
    total_paper_count = 0

    for record in responses:
        papers = record['papers']  # 获取所有论文
        author_name = record['a']

        # 统计部门内所有论文的总数
        total_paper_count += len(papers)
        author_set.add(author_name)
        
        # 只返回前8篇论文
        limited_papers = papers[:8]
        
        # 遍历每篇论文并将其添加到结果列表
        # for paper in papers:
        for paper in limited_papers:
            formatted_results.append({
                "paper": paper.get('name'),
                "author": author_name.get('name') ,
                "venue": paper.get('source'),
                "citation": paper.get('citation'),
                "keywords": paper.get('keywords'),  # 假设关键词是一个单一值或字符串
                "year": paper.get("year")
            })

    return formatted_results

def format_department_keyword_response(responses):
    formatted_results = []

    for record in responses:
        papers = record['papers']  # 获取论文信息
        author_name = record['a']
        formatted_results.append({
            "paper": papers.get('name'),
            "venue": papers.get('source'),
            "year": papers.get('year'),
            "author": author_name.get('name'),
            "citation": papers.get('citation'),
            "keywords": papers.get('keywords') 
        })

    return formatted_results

def format_department_venue_response(responses):
    formatted_results = []

    for record in responses:
        papers = record['papers']  # 获取论文信息
        author_name = record['a']
        formatted_results.append({
            "paper": papers.get('name'),
            "year": papers.get('year'),
            "author": author_name.get('name'),
            "citation": papers.get('citation'),
            "keywords": papers.get('keywords') 
        })

    return formatted_results

def format_department_department_response(responses):
    formatted_results = []

    for record in responses:
        # 遍历每篇论文
        for paper in record['papers']:
            formatted_results.append({
                "paper": paper.get('name'),
                "venue": paper.get('source'),
                "year": paper.get('year'),
                "citation": paper.get('citation'),
                "abstract": paper.get('abstract'),
                "keywords": paper.get('keywords')  # 假设关键词是单一值或字符串
            })

    return formatted_results

def format_keyword_year_response(responses):
    formatted_results = []

    for record in responses:
        papers = record['papers']  # 获取论文信息
        author_name = record['a']
        formatted_results.append({
            "paper": papers.get('name'),
            "author": author_name.get('name'),
            "venue": papers.get('source'),
            "citation": papers.get('citation'),
            "keywords": papers.get('keywords')
        })

    return formatted_results
def format_keyword_venue_response(responses):
    formatted_results = []

    for record in responses:
        papers = record['papers']  # 获取论文信息
        author_name = record['a']
        formatted_results.append({
            "paper": papers.get('name'),
            "year": papers.get('year'),
            "author": author_name.get('name'),
            "citation": papers.get('citation'),
            "keywords": papers.get('keywords')
        })

    return formatted_results

def format_year_venue_response(responses):
    formatted_results = []

    for record in responses:
        papers = record['papers']  # 获取论文信息
        author_name = record['a']
        formatted_results.append({
            "paper": papers.get('name'),
            "author": author_name.get('name'),
            "citation": papers.get('citation'),
            "keywords": papers.get('keywords')
        })

    return formatted_results

def format_combined_response(combined_responses, inferred_types):
    authors = get_key(inferred_types, 'Author') #判断是否有多个作者
    departments = get_key(inferred_types, 'Department') #判断是否有多个部门
    if "Author" in inferred_types.values() and "Year" in inferred_types.values():
        return format_author_year_response(combined_responses)
    elif "Author" in inferred_types.values() and "Keyword" in inferred_types.values():
        return format_author_keyword_response(combined_responses)
    elif "Author" in inferred_types.values() and "Venue" in inferred_types.values():
        return format_author_venue_response(combined_responses)
    

    elif "Department" in inferred_types.values() and "Year" in inferred_types.values():
        return format_department_year_response(combined_responses)
    elif "Department" in inferred_types.values() and "Keyword" in inferred_types.values():
        return format_department_keyword_response(combined_responses)
    elif "Department" in inferred_types.values() and "Venue" in inferred_types.values():
        return format_department_venue_response(combined_responses)

    elif "Keyword" in inferred_types.values() and "Year" in inferred_types.values():
        return format_keyword_year_response(combined_responses)
    elif "Keyword" in inferred_types.values() and "Venue" in inferred_types.values():
        return format_keyword_venue_response(combined_responses)

    elif "Year" in inferred_types.values() and "Venue" in inferred_types.values():
        return format_year_venue_response(combined_responses)
    
    elif len(authors)> 1:
        return format_author_author_response(combined_responses)
    elif len(departments)> 1:
        return format_department_department_response(combined_responses)
    # 添加其他组合的格式化逻辑...
    
    return []

import json

from django.db.models import Q
from django.shortcuts import render, HttpResponse
from django.views.decorators.csrf import csrf_exempt

from app01.models import PublicationDatasets
from app01.views.view_knowledge_graph import flite_years, query_departments_chart, query_departments_min_chart, \
    flite_dept, query_paper_department, query_paper_department_year, query_keywords, query_find_paper_by_keyword
from app01.views.view_knowledge_graph import query_author, query_year, query_authors_chart, flite_authors, \
    update_data_for_wordcloud_dept, update_data_for_chart_dept, update_data_for_treemap_dept, \
    update_data_for_donut_dept, update_data_for_donut_author, update_data_for_chart_author, \
    update_data_for_wordcloud_author, update_data_for_treemap_author


# Create your views here.

def index(request):
    # publication_list = get_publications_datasets()
    # publication_list = query_publication()
    with open('app01/datasets/Init.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    publication_list = list(data)
    return render(request, "index.html", {"publication_list": publication_list})


def get_data_for_chart(request):
    paper_year = query_year()
    paper_department = query_paper_department()
    paper_department_year = query_paper_department_year()
    paper_author = query_authors_chart()
    results = {
        'chart1': paper_year,
        'donut': paper_department,
        'chart_d_y': paper_department_year,
        'heatmap': paper_author,
    }
    return HttpResponse(json.dumps(results))


def get_keyword_for_word_cloud(request):
    results = query_keywords()
    return HttpResponse(json.dumps(results))


@csrf_exempt
def keyword_find_paper(request):
    data = json.loads(request.body)
    keyword = data['keyword']
    paper_list = query_find_paper_by_keyword(keyword)
    return HttpResponse(json.dumps(paper_list))


@csrf_exempt
def update_for_chart(request):
    new_donut = []
    new_chart_d_y = []
    new_treemap = []
    data = json.loads(request.body)
    name = data['name']
    group = data['group']
    print(data)
    if group == 4:
        new_donut = update_data_for_donut_author(name)
        new_chart_d_y = update_data_for_chart_author(name)
        new_treemap = update_data_for_treemap_author(name)

    elif group == 1:
        new_donut = update_data_for_donut_dept(name)
        new_chart_d_y = update_data_for_chart_dept(name)
        new_treemap = update_data_for_treemap_dept(name)

    return HttpResponse(json.dumps({
        'donut': new_donut,
        'chart_d_y': new_chart_d_y,
        'treemap': new_treemap,
    }))


def get_years_from_neo4j(request):
    print('Get years from neo4j')
    years_publications = query_year()
    return HttpResponse(json.dumps(years_publications))


def get_authors_from_neo4j(request):
    print('Get authors from neo4j')
    authors_publications = query_authors_chart()
    return HttpResponse(json.dumps(authors_publications))


def get_departments_from_neo4j(request):
    print('Get departments from neo4j')
    departments_authors = query_departments_chart()
    return HttpResponse(json.dumps(departments_authors))


@csrf_exempt
def get_departments_min_from_neo4j(request):
    data = json.loads(request.body)
    min = data['min']
    departments_authors_min = query_departments_min_chart(min)
    return HttpResponse(json.dumps(departments_authors_min))


def get_years(request):
    print('Getting years')

    years_list = sorted(list(PublicationDatasets.objects.values_list('PublishedYears', flat=True).distinct()))
    years_publications = get_num_of_years_publications(years_list)

    print('Finished getting years')
    return HttpResponse(json.dumps(years_publications))


def get_authors(request):
    print("Getting authors...")

    authors_list = query_author()
    authors_publications = get_num_of_authors_publications(authors_list)

    print('Finished getting authors.')
    return HttpResponse(json.dumps(authors_publications))


def get_keywords(request):
    print("Getting keywords...")

    keyword_list = get_each_keywords()
    keywords_publications = get_num_of_keywords_publications(keyword_list)

    print("Finished getting keywords")
    return HttpResponse(json.dumps({'keywords_list': keyword_list, 'keywords_publications': keywords_publications}))


def get_series(request):
    print("Getting series...")

    series_list = get_each_series()
    series_publications = get_num_of_series_publications(series_list)

    print("Finished getting series")
    return HttpResponse(json.dumps({'series_list': series_list, 'series_publications': series_publications}))


@csrf_exempt
def query(request):
    print('Getting query')

    # all_IdName = list(PublicationDatasets.objects.all().values_list('IdName', flat=True))
    with open('app01/datasets/All_id.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    all_IdName = list(data)

    data = json.loads(request.body)

    keywords_results = []
    series_results = []
    year_results = []

    # 检索选中的作者
    author_results = flite_authors(data['author'])
    year_results = flite_years(data['year'])
    dept_results = flite_dept(data['dept'])

    # 检索选中的作者
    # for i in range(len(data['author'])):
    #     a_result = list(PublicationDatasets.objects.filter((Q(AuthorOne=data['author'][i]) | Q(
    #         AuthorTwo=data['author'][i]) | Q(AuthorThree=data['author'][i]) | Q(AuthorFour=data['author'][i]) | Q(
    #         AuthorFive=data['author'][i]) | Q(AuthorSix=data['author'][i]) | Q(AuthorSeven=data['author'][i]) | Q(
    #         AuthorEight=data['author'][i]) | Q(AuthorNine=data['author'][i]) | Q(
    #         AuthorTen=data['author'][i]))).values_list('IdName', flat=True))
    #     if i == 0:
    #         author_results = a_result
    #     else:
    #         author_results = list(set(a_result) & set(author_results))

    # 检索选中的关键词
    for i in range(len(data['keyword'])):
        b_result = list(
            PublicationDatasets.objects.filter(Keyword__icontains=data['keyword'][i]).values_list('IdName', flat=True))
        if i == 0:
            keywords_results = b_result
        else:
            keywords_results = list(set(keywords_results) & set(b_result))

    # 检索选中的series
    for i in range(len(data['series'])):
        c_result = list(PublicationDatasets.objects.filter(Sources=data['series'][i]).values_list('IdName', flat=True))
        if i == 0:
            series_results = c_result
        else:
            series_results = list(set(series_results) & set(c_result))

    # 检索选中的年份
    # for i in range(len(data['year'])):
    #     d_result = list(
    #         PublicationDatasets.objects.filter(PublishedYears=data['year'][i]).values_list('IdName', flat=True))
    #     if i == 0:
    #         year_results = d_result
    #     else:
    #         year_results = list(set(year_results) & set(d_result))

    results = common_elements(author_results, keywords_results, series_results, year_results, dept_results)
    print('Finished getting query')
    return HttpResponse(json.dumps({'all_IdName': all_IdName, 'result_IdName': results}))


def get_publications_datasets():
    print("Getting publications datasets...")
    publications = PublicationDatasets.objects.all()
    publications_list = []
    for publication in publications:
        publication_dist = {}
        publication_dist.update({'Name': publication.Name})
        publication_dist.update({'Authors': [publication.AuthorOne, publication.AuthorTwo, publication.AuthorThree,
                                             publication.AuthorFour, publication.AuthorFive, publication.AuthorSix,
                                             publication.AuthorSeven, publication.AuthorEight,
                                             publication.AuthorNine, publication.AuthorTen]})
        publication_dist.update({'Sources': publication.Sources})
        publication_dist.update({'PublishedYears': publication.PublishedYears})
        publication_dist.update({'Doi': publication.Doi})
        publication_dist.update({'Abstracts': publication.Abstracts})
        publication_dist.update({'IdName': publication.IdName})
        publication_dist.update({'Citation': publication.Citation})
        publication_dist.update({'Keywords': publication.Keyword})
        publications_list.append(publication_dist)
    print('Finished getting publications datasets.')
    return publications_list


def get_each_series():
    print("Getting series...")

    series_list = list(PublicationDatasets.objects.values_list('Sources', flat=True).distinct())

    print("Finished getting series")
    return series_list


def get_each_keywords():
    print("Getting keywords...")
    keyword_list = []

    keyword_all = list(PublicationDatasets.objects.values_list('Keyword', flat=True).distinct())
    keyword_all.remove("")
    for keyword in keyword_all:
        temp = keyword.lower()
        temp = temp.split(', ')
        for each in temp:
            keyword_list.append(each)
    keyword_list = list(set(keyword_list))
    print("Finished getting keywords")
    get_num_of_keywords_publications(keyword_list)
    return keyword_list


def get_num_of_series_publications(series_list):
    print("Getting number of publications...")

    series_publications_dist = {}
    for series in series_list:
        count = PublicationDatasets.objects.filter(Sources=series).count()
        series_publications_dist.update({series: count})
    series_publications_dist = {key: series_publications_dist[key] for key in
                                sorted(series_publications_dist.keys(), key=series_publications_dist.get, reverse=True)}

    print("Finished getting number of publications")
    return series_publications_dist


def get_num_of_years_publications(years_list):
    print('Getting number of years')

    years_publications_dist = []
    for year in years_list:
        count = PublicationDatasets.objects.filter(PublishedYears=year).count()
        years_publications_dist.append({
            'group': year,
            'value': count
        })

    print('Finished getting number of years')
    return years_publications_dist


def get_num_of_keywords_publications(keywords_list):
    print("Getting number of keywords...")
    keywords_publications_dist = {}
    for keyword in keywords_list:
        item_count = PublicationDatasets.objects.filter(Keyword__icontains=keyword).count()
        keywords_publications_dist.update({keyword: item_count})
    keywords_publications_dist = {key: keywords_publications_dist[key] for key in
                                  sorted(keywords_publications_dist.keys(), key=keywords_publications_dist.get,
                                         reverse=True)}
    print("Finished getting number of keywords")
    return keywords_publications_dist


def get_each_authors():
    author1 = list(PublicationDatasets.objects.exclude(AuthorOne=None)
                   .values_list('AuthorOne', flat=True).distinct())
    author2 = list(PublicationDatasets.objects.exclude(AuthorTwo=None)
                   .values_list('AuthorTwo', flat=True).distinct())
    author3 = list(PublicationDatasets.objects.exclude(AuthorThree=None)
                   .values_list('AuthorThree', flat=True).distinct())
    author4 = list(PublicationDatasets.objects.exclude(AuthorFour=None)
                   .values_list('AuthorFour', flat=True).distinct())
    author5 = list(PublicationDatasets.objects.exclude(AuthorFive=None)
                   .values_list('AuthorFive', flat=True).distinct())
    author6 = list(PublicationDatasets.objects.exclude(AuthorSix=None)
                   .values_list('AuthorSix', flat=True).distinct())
    author7 = list(PublicationDatasets.objects.exclude(AuthorSeven=None)
                   .values_list('AuthorSeven', flat=True).distinct())
    author8 = list(PublicationDatasets.objects.exclude(AuthorEight=None)
                   .values_list('AuthorEight', flat=True).distinct())
    author9 = list(PublicationDatasets.objects.exclude(AuthorNine=None)
                   .values_list('AuthorNine', flat=True).distinct())
    author10 = list(PublicationDatasets.objects.exclude(AuthorTen=None)
                    .values_list('AuthorTen', flat=True).distinct())

    authors_list = author1 + author2 + author3 + author4 + author5 + author6 + author7 + author8 + author9 + author10
    authors_list = list(set(authors_list))
    return authors_list


def get_num_of_authors_publications(authors_list):
    print("Getting authors publications...")

    author_publication_dist = []
    for author in authors_list:
        active_item_count = PublicationDatasets.objects.filter(
            Q(AuthorOne=author) | Q(AuthorTwo=author) | Q(AuthorThree=author) | Q(AuthorFour=author) | Q(
                AuthorFive=author) | Q(AuthorSix=author) | Q(AuthorSeven=author) | Q(AuthorEight=author) | Q(
                AuthorNine=author) | Q(AuthorTen=author)).count()
        author_publication_dist.append({
            'group': author,
            'value': active_item_count
        })
    # author_publication_dist = {key: author_publication_dist[key] for key in
    #                            sorted(author_publication_dist.keys(), key=author_publication_dist.get, reverse=True)}

    print("Finished getting authors publications.")
    return author_publication_dist


# 过滤空list
def common_elements(*lists):
    # 过滤掉空列表
    non_empty_lists = [lst for lst in lists if lst]

    # 如果过滤后没有列表，返回空列表
    if not non_empty_lists:
        return []

    # 使用集合的交集操作找出共同的元素
    common_set = set(non_empty_lists[0])
    for lst in non_empty_lists[1:]:
        common_set &= set(lst)

    return list(common_set)



"""
URL configuration for django_HKUST project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path

from app01.views import views, view_knowledge_graph, view_chart, view_qa

urlpatterns = [
    # 页面
    path('', views.index),
    path('KG/', view_knowledge_graph.knowledge_graph),
    path('chart/', view_chart.chart),
    path('qa/', view_qa.qa),

    # 测试
    path('test/query/', view_knowledge_graph.query_base),
    path('test/query_paper/', view_knowledge_graph.query_paper),

    # 获取数据_base
    # 1.csv_data
    path('data/authors', views.get_authors),
    path('data/years', views.get_years),
    path('data/keywords', views.get_keywords),
    path('data/series', views.get_series),
    path('query/result', views.query),

    # 2.graph_data
    path('data_neo4j/years', views.get_years_from_neo4j),
    path('data_neo4j/authors', views.get_authors_from_neo4j),
    path('data_neo4j/departments', views.get_departments_from_neo4j),
    path('data_neo4j/departmentsMin', views.get_departments_min_from_neo4j),
    path('data_neo4j/forChart', views.get_data_for_chart),
    path('data_neo4j/keyword', views.get_keyword_for_word_cloud),
    path('data_neo4j/keywordFindPaper', views.keyword_find_paper),
    path('data_neo4j/forchartupdate', views.update_for_chart),

    # 获取数据_graph
    path('graph/query/', view_knowledge_graph.query_base),
    path('graph/query_paper/', view_knowledge_graph.query_paper),
    path('graph/query_all/', view_knowledge_graph.query_all),

    # 获取数据_qa
    path('qa/chat', view_qa.chat),
]

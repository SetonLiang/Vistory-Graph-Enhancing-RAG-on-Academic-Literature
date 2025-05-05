# Vistory-Graph-Enhancing-RAG-on-Academic-Literature

## Install

- Install Neo4j Database
- Install Langchain
- Install Django
- Set Environment Variable

  ```
  os.environ["OPENAI_API_BASE"]
  os.environ["OPENAI_API_KEY"] 
  ```

  

## Prepare the Dataset

1. Crawling the Literature data

   1. crawl_google22.py: 输入老师的google scholar的个人主页url来下载csv文件

   2. crawl_google.py: 扩充csv文件的ID和Citation列
      1. read_name(): 将csv文件的Name列导入到txt文件
      2. extract_href(): 遍历txt内容得到bib.txt和links.txt, 并在csv文件添加Citation列，同时得到作者名_pdf.json文件，记录论文的pdf链接
      3. open_txtfile(): 打开bib文件，将ID写入csv文件

      > 记得改路径和文件名，对应每个老师；每次执行整个py文件的时候要把bib和links文件删除

   3.  crawl_pdf.py: 下载论文的pdf文件
       1.  download_pdf(): 根据作者名_pdf.json文件下载对应pdf
       2.  find_notcrawl_pdf(): 如果有缺失的pdf，执行该函数帮你查找
         > 作者名_pdf.json，每次只用改作者名；对于IEEE的pdf需要手动点击下载，对于Arxiv等pdf，需要手动更改文件名
         > 文件名为论文题目

   4. crawl_keywords.py: 得到论文的关键词
      1. find_pdfs(): 将pdf的keywords存为json
      2. save_json(): 将json的keywords存入csv

   5. crawl_img.py: 得到论文的图片

      > 存的图片直接丢到一起，有重复直接跳过

2. Updating Neo4j Database

   1. load_data.py: 将csv文件上传到图数据库中

      > 文件格式: 部门文件夹-老师.csv (utf-8, 不能有空白格)

   2. create_entity_json.py

      1. fetch_all_nodes_and_labels(): 获取所有节点和标签的映射关系
      2. save_map_to_json(): 获取每个节点的信息和embedding

## How to run this system
Enter django_HKUST directory and excecute the following scrpits:
```
python manage.py runserver
```

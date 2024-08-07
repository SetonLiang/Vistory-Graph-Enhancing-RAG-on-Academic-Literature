from random import choice
import requests  # 用于对url发起响应
import io, csv, re,os
from bs4 import BeautifulSoup
import pandas as pd
from Crypto.Cipher import AES  # 用于加密函数中的AES加密操作
from base64 import b64encode  # 用于字符串进行base64编码，并返回字节
import json  # 用于将data转化为字符串模式
import time
from selenium import webdriver
from urllib.request import urlopen,Request

url1 = 'https://repository.hkust.edu.hk/ir/Search/Results?lookfor=Zhou+jinni&type=AllFields&filter%5B%5D=author_facet%3A%22Zhou%2C+Jinni%22'
ua = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36 Edg/116.0.1938.62",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36 Edg/93.0.961.52"
]
headers = {
    # 'Host': 'ieeexplore.ieee.org',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
    "Referer":"https://ieeexplore.ieee.org/",
    'Accept': 'application/pdf'
}

resp1 = requests.get(url1,headers=headers)
resp1.encoding="utf-8"
content = resp1.text
content = content.replace('<br>', '')
soup = BeautifulSoup(content, 'html.parser')
pub_pages = soup.find('ul', class_='pagination')
resp1.close()

pages = []
titles = []
venues = []
publish_years = []
all_authors = []
sources = []
doi_s = []
abstracts = []
keywords = []
# pdf_s = []
pubs = {}  # 用于绑定论文名称和作者
flag = 0  # 用于记录每篇论文的作者

# for page in pub_pages:
#     effect_page = page.string
#     total_page = page
#     # print(effect_page)
#     if effect_page == 'Next »':
#         continue
#     if '[' in effect_page:
#         num_pages = int(str(effect_page).replace('[', '').replace(']', ''))
#         print(num_pages)
#         for num in range(1, num_pages+1): pages.append(num)
#     if '[' not in effect_page:
#         pages.append(1)
#     # if '\n' not in effect_page:
#     #     pages.append(effect_page)
#     # if effect_page == None:
#     #     begin = str(total_page).find('>[')+2
#     #     end = str(total_page).find(']<')
#     #     num_pages = int(str(total_page)[begin:end])
#     #     for num in range(1, num_pages + 1): pages.append(num)
# print(pages)
#
# for num in pages:
#     # if int(num) == 2: break
#     # if int(num) < 11: continue
#     #     url = 'https://repository.hkust.edu.hk/ir/Search/Results?filter%5B%5D=author_facet%3A%22Zeng%2C+Wei%22&lookfor=wei+zeng&type=AllFields'
#     # else:
#     url = f'https://repository.hkust.edu.hk/ir/Search/Results?lookfor=Zhou+jinni&type=AllFields&filter%5B%5D=author_facet%3A%22Zhou%2C+Jinni%22&page={int(num)}'
#     resp = requests.get(url, headers=headers)
#     resp.encoding = "utf-8"
#     content = resp.text
#     content = content.replace('<br>', '')
#     soup = BeautifulSoup(content, 'html.parser')
#     # print(soup)
#     pub_pages = soup.find('ul', class_='pagination')
#     pub_titles = soup.findAll('a', attrs={'class': 'title'})
#     pub = soup.findAll('div', class_='middle resultitem')
#
#     for i in pub_titles:
#         # 获取论文title
#         effect_i = str(i)[str(i).find('>')+1:str(i).rfind('</a>')].strip()
#         print(effect_i)
#         titles.append(effect_i)
#         sub_url = f"https://repository.hkust.edu.hk{i.get('href')}"
#         print(num, sub_url)
#         """
#             爬取子页面的内容
#         """
#         sub_resp = requests.get(sub_url, headers=headers)
#         sub_resp.encoding = 'utf-8'
#         content2 = sub_resp.text
#         soup2 = BeautifulSoup(content2, 'html.parser')
#         # print(soup2)
#
#         # 获取DOI
#         abstract_len = len(abstracts)
#         ori_abstract = soup2.find('table', attrs={'class': 'table table-striped', 'id': 'pubDetailsTable'})
#         ori_doi = soup2.find(attrs={"name": "Doi"})['content']
#         for j in ori_abstract:
#             if "<th>Abstract</th>" in str(j):
#                 abstracts.append(j.find('td').string)
#         # abstract = soup2.find(attrs={"name": "DCTERMS.abstract"})['content']  # 有些没有摘要
#         print(abstract_len,len(abstracts))
#         if abstract_len == len(abstracts):
#             abstracts.append('None')
#         doi = soup2.find('table', attrs={'class': 'table table-striped', 'id': 'pubDetailsTable'}).findAll('a')
#         new_doi = ''
#         for each_doi in doi:
#             if str(ori_doi) in each_doi:
#                 new_doi = each_doi.get('href')
#         doi_s.append(new_doi)
#         time.sleep(1)
#
#     # print(titles, len(titles))
#
#     for j in pub:
#         effect_other = str(j)
#         if 'Article' in effect_other:
#             effect_other_begin = effect_other.find('Article')
#             effect_other_name = effect_other[effect_other_begin:]
#             final_name = effect_other_name.split(',')[0]
#             if '\n' in final_name:
#                 final_name = final_name[final_name.rfind('\n'):].replace('\n','').split()
#                 final_name = ' '.join(final_name)
#             # print(final_name)
#             venues.append(final_name)
#             publish_years.append(effect_other_name.split(',')[-1].replace('</div>', '').replace('\n', '').split()[0])
#         elif 'Conference paper' in effect_other:
#             effect_other_begin = effect_other.find('Conference paper')
#             effect_other_name = effect_other[effect_other_begin:]
#             venues.append(effect_other_name.split(',')[0])
#             publish_years.append(effect_other_name.split(',')[-1].replace('</div>', '').replace('\n', '').split()[0])
#         elif 'Book chapter' in effect_other:
#             effect_other_begin = effect_other.find('Book chapter')
#             effect_other_name = effect_other[effect_other_begin:]
#             venues.append(effect_other_name.split(',')[0])
#             publish_years.append(effect_other_name.split(',')[-1].replace('</div>', '').replace('\n', '').split()[0])
#         elif 'Thesis' in effect_other:
#             effect_other_begin = effect_other.find('Thesis')
#             effect_other_name = effect_other[effect_other_begin:]
#             venues.append(effect_other_name.split(',')[0])
#             publish_years.append(effect_other_name.split(',')[-1].replace('</div>', '').replace('\n', '').split()[0])
#         elif 'Dataset' in effect_other:
#             effect_other_begin = effect_other.find('Dataset')
#             effect_other_name = effect_other[effect_other_begin:]
#             venues.append(effect_other_name.split(',')[0])
#             publish_years.append(effect_other_name.split(',')[-1].replace('</div>', '').replace('\n', '').split()[0])
#         elif 'Book' in effect_other:
#             effect_other_begin = effect_other.find('Book')
#             effect_other_name = effect_other[effect_other_begin:]
#             venues.append(effect_other_name.split(',')[0])
#             publish_years.append(effect_other_name.split(',')[-1].replace('</div>', '').replace('\n', '').split()[0])
#         elif 'Preprint' in effect_other:
#             effect_other_begin = effect_other.find('Preprint')
#             effect_other_name = effect_other[effect_other_begin:]
#             venues.append(effect_other_name.split(',')[0])
#             publish_years.append(effect_other_name.split(',')[-1].replace('</div>', '').replace('\n', '').split()[0])
#         elif 'Patent' in effect_other:
#             effect_other_begin = effect_other.find('Patent')
#             effect_other_name = effect_other[effect_other_begin:]
#             venues.append(effect_other_name.split(',')[0])
#             publish_years.append(effect_other_name.split(',')[-1].replace('</div>', '').replace('\n', '').split()[0])
#         elif 'Technical report' in effect_other:
#             effect_other_begin = effect_other.find('Technical report')
#             effect_other_name = effect_other[effect_other_begin:]
#             venues.append(effect_other_name.split(',')[0])
#             publish_years.append(effect_other_name.split(',')[-1].replace('</div>', '').replace('\n', '').split()[0])
#         elif 'Research report' in effect_other:
#             effect_other_begin = effect_other.find('Research report')
#             effect_other_name = effect_other[effect_other_begin:]
#             venues.append(effect_other_name.split(',')[0])
#             publish_years.append(effect_other_name.split(',')[-1].replace('</div>', '').replace('\n', '').split()[0])
#         elif 'Other' in effect_other:
#             effect_other_begin = effect_other.find('Other')
#             effect_other_name = effect_other[effect_other_begin:]
#             venues.append(effect_other_name.split(',')[0])
#             publish_years.append(effect_other_name.split(',')[-1].replace('</div>', '').replace('\n', '').split()[0])
#
#         effect_author = j.findAll('a')
#         authors = []
#         author = ''
#         source = ''
#         length = len(sources)
#         # print(effect_author)
#         for name in effect_author:
#             orgin = name
#             name = str(name)
#             # if 'type=Author' in name:
#             #     begin = 'type=Author">'
#             #     begin_idx = name.find(begin, ) + len(begin)
#             #     end_idx = name.rfind('</a>')
#             #     author = name[begin_idx:end_idx]
#             #     # print(author)
#             #     authors.append(author)
#             if 'author_facet' in name:
#                 authors.append(orgin.string)
#             elif 'type=JournalTitle' in name:
#                 # begin = 'type=JournalTitle">'
#                 # begin_idx = name.find(begin, ) + len(begin)
#                 # end_idx = name.rfind('</a>')
#                 # source = name[begin_idx:end_idx]
#                 # sources.append(source)
#                 sources.append(orgin.string)
#         if length == len(sources):
#             sources.append('None')
#
#         # pubs[titles[flag]] = authors
#         # flag += 1
#         all_authors.append(authors)
#         # break
#     resp.close()
#
# # for k in doi_s:
# #     if k != '':
# #         print(k)
# #         resp3 = requests.get(k, headers=headers)
# #         resp3.encoding = 'utf-8'
# #         content3 = resp3.text
# #         soup3 = BeautifulSoup(content3, 'html.parser')
# #         script_tags = soup3.find_all("script")
# #         # print(script_tags)
# #
# #         # 提取<script>标签中的JavaScript代码
# #         javascript_code = []
# #         for script_tag in script_tags:
# #             code = str(script_tag)
# #             if code:
# #                 javascript_code.append(code)
# #         # print(javascript_code)
# #         # 使用正则表达式匹配变量值
# #         pattern = r"xplGlobal\.document\.metadata\s*=\s*(.*?)(?:;|$)"
# #         for code in javascript_code:
# #             matches = re.findall(pattern, code)
# #             # print(matches)
# #             if matches:
# #                 abstract_begin = matches[0].rfind('"abstract":')
# #                 abstract_end = matches[0][abstract_begin + len('"abstract":'):].find(',"')
# #                 abstract = matches[0][abstract_begin + len('"abstract":'):][:abstract_end]
# #                 # print(abstract)
# #                 abstracts.append(abstract)
# #                 pdf_begin = matches[0].find('"pdfUrl":')
# #                 pdf_end = matches[0][pdf_begin:].find(',')
# #                 pdf = matches[0][pdf_begin:][:pdf_end]
# #                 pdf = pdf[10:-1]
# #                 print(pdf)
# #                 pdf_s.append(pdf)
# #                 keyword_begin = matches[0].find('"Author Keywords","kwd":[')
# #                 keyword_end = matches[0][keyword_begin + len('"Author Keywords","kwd":['):].find(']}]')
# #                 keyword = matches[0][keyword_begin + len('"Author Keywords","kwd":['):][:keyword_end]
# #                 print(keyword)
# #                 keywords.append(keyword)
# #         time.sleep(1)
#
# print(abstracts, len(abstracts))
# # print(keywords, len(keywords))
# print(all_authors, len(all_authors))
# print(sources, len(sources))
# print(venues, len(venues))
# print(publish_years, len(publish_years))
# print(doi_s, len(doi_s))
# pubs['Name'] = titles
# pubs['Authors'] = all_authors
# pubs['sources'] = sources
# pubs['venues'] = venues
# pubs['Published year'] = publish_years
# pubs['Doi'] = doi_s
# pubs['Abstract'] = abstracts
# # pubs['Keywords'] = keywords
# df = pd.DataFrame(pubs)
# print(df)
# df['Authors'] = df['Authors'].astype(str).str.replace('[', '').str.replace(']', '')  # 去掉作者单元格的[]
# df.to_csv('zhoujinni'+'.csv', index=False)

# 下载对应的pdf文件
def crawl_pdf(file):
    df = pd.read_csv(file)
    pdf_s = []
    # 获取特定列（在这个例子中是"Name"列）
    names = df['Doi']
    i = 0
    for name in names:
        if i < 31:
            i+=1
            continue

        print(name)
        url_pdf = name
        pdf = ''
        """
            爬取子页面的内容
        """
        resp_pdf = requests.get(url_pdf, headers=headers)
        resp_pdf.encoding = 'utf-8'
        content4 = resp_pdf.text
        soup4 = BeautifulSoup(content4, 'html.parser')
        script_tags = soup4.find_all("script")
        # print(script_tags)

        # 提取<script>标签中的JavaScript代码
        javascript_code = []
        for script_tag in script_tags:
            code = script_tag.string
            if code:
                javascript_code.append(code)
        # print(javascript_code)
        # 使用正则表达式匹配变量值
        pattern = r"xplGlobal\.document\.metadata\s*=\s*(.*?)(?:;|$)"
        variables = {'js_video_url': None}
        for code in javascript_code:
            matches = re.findall(pattern, code)
            # print(matches)
            if matches:
                pdf_begin = matches[0].find('"pdfUrl":')
                pdf_end = matches[0][pdf_begin:].find(',')
                pdf = matches[0][pdf_begin:][:pdf_end]
                pdf = pdf[10:-1]
                print(pdf)
                print(1)
                pdf_s.append(pdf)
        import time
        time.sleep(1)

    json_data = json.dumps(pdf_s, ensure_ascii=False, indent=4)

    # 将 json 数据写入文件
    with open("zengwei2.json", "w", encoding='utf-8') as json_file:
        json_file.write(json_data)


# def download_pdf2(json_file):
#     with open(json_file, "r", encoding='utf-8') as f:
#         pdf_s = json.load(f)
#         i = 1
#         for pdf in pdf_s:
#             if pdf == '': break
#             new_url_pdf = f"https://ieeexplore.ieee.org{pdf}"
#             resp_pdf2 = requests.get(new_url_pdf, headers=headers)
#             resp_pdf2.encoding = 'utf-8'
#             content5 = resp_pdf2.text
#             soup5 = BeautifulSoup(content5, 'html.parser')
#             download_pdf_url = soup5.findAll('iframe')
#             print(new_url_pdf)
#             for item in download_pdf_url:
#                 item2 = str(item)
#                 effect_item = item2[item2.find('src="')+len('src="'):item2.rfind('">')].replace('amp;', '')
#                 print(effect_item)
#             try:
#                 # 发送GET请求
#                 response = requests.get(effect_item, headers=headers)
#                 # print(response.content)
#                 with open(f'{i}.pdf', 'wb+') as f:
#                     f.write(response.content)
#                 i += 1
#                 # 停一下防禁ip
#                 import time
#                 time.sleep(1)
#
#             except requests.exceptions.RequestException as e:
#                 print(f"请求发生错误: {e}")
#
#             except IOError as e:
#                 print(f"无法写入文件: {e}")
def download_pdf(json_file):
    with open(json_file, "r", encoding='utf-8') as f:
        pdf_s = json.load(f)
        i=0
        for name,url in pdf_s.items():
            print(name,url)
            # ret = Request(url,headers=headers)
            # u = urlopen(ret)
            # f = open(f"{name}.pdf", 'wb')
            #
            # block_sz = 8192
            # while True:
            #     buffer = u.read(block_sz)
            #     if not buffer:
            #         break
            #     f.write(buffer)
            # f.close()
            i += 1
            # if i<=5:continue
            # elif i==20:break
            down_load_dir = os.path.abspath("G://VIStory/project/hkust_papers/AI/liangjunwei")  # 浏览器会自动创建文件夹 写绝对路径
            options = webdriver.ChromeOptions()
            options.add_experimental_option("excludeSwitches", ['enable-automation'])
            prefs = {
                "download.default_directory": down_load_dir,
                "download.prompt_for_download": False,
                "download.directory_upgrade": True,
                "plugins.always_open_pdf_externally": True
            }
            options.add_experimental_option('prefs', prefs)
            driver = webdriver.Chrome(options=options)
            driver.get(url)
            pdf_content = driver.page_source
            with open(f'{name}.pdf', 'wb') as file:
                file.write(pdf_content.encode('utf-8'))
            time.sleep(25)

        driver.quit()

def find_notcrawl_pdf(folder,jsonfile):
    df = pd.read_csv(folder, encoding='gbk')
    paper_name = df['Name'].values
    with open(jsonfile,'r') as file:
        pdf = list(json.load(file).keys())
    # 找出在lines但不在pdf列表中的值
    not_crawl_pdfs = [line for line in paper_name if line not in pdf]

    return not_crawl_pdfs
file = 'src/huxuming.csv'
# crawl_pdf(file)
json_file = 'huxuming_pdf.json'
# download_pdf(json_file)


txtfile = 'G:\VIStory\project\src\Google_scholar\scholars.txt'
# jsonfile = 'G:\VIStory\project\src/huxuming_pdf.json'
jsonfile = "G:\VIStory\project\hkust_papers\CMA\\fanmingming\\fanmingming_pdf.json"
# folder = "G://VIStory/project/src/huxuming.csv"
folder = "G:\VIStory\project\hkust_papers\CMA\All\\fanmingming.csv"
res = find_notcrawl_pdf(folder,jsonfile)
print(res,len(res))
"""
以下是下载pdf
"""
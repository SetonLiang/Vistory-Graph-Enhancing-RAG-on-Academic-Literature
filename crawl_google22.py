import time
import re
import pandas as pd
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

# 请求用户输入Google Scholar主页的网址
url_base = input("请输入Google Scholar主页的网址: ")

# Selenium的Chrome驱动路径
chrome_driver_path = 'G:\VIStory\project\src\Google_scholar\chromedriver.exe'

# 设置Chrome选项
chrome_options = Options()
chrome_options.add_argument('--headless')  # 运行时不显示浏览器
chrome_options.add_argument('--disable-gpu')
chrome_options.add_argument('--window-size=1920x1080')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

# 初始化Chrome驱动
driver = webdriver.Chrome(service=Service(executable_path='G:\VIStory\project\src\Google_scholar\chromedriver.exe',options=chrome_options))

def get_soup(url, retries=5, initial_delay=5, backoff_factor=2):
    """
    发送HTTP请求到指定URL并获取BeautifulSoup对象。

    参数:
    - url: 要请求的网页的URL字符串。

    返回:
    - BeautifulSoup对象，用于进一步解析网页内容。

    异常:
    - HTTPError: 如果响应的状态码不是200，即请求失败。
    """
    delay = initial_delay

    for attempt in range(retries):
        try:
            if attempt > 0:
                print(f"Waiting for {delay} seconds before retrying...")
                time.sleep(delay)
                delay *= backoff_factor

            driver.get(url)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, 'gsc_a_tr'))
            )
            html = driver.page_source
            return BeautifulSoup(html, 'html.parser')
        except Exception as e:
            print(f"Request failed: {e}")
            if attempt == retries - 1:
                raise

    raise Exception("Max retries exceeded")

def get_abstract(url):
    """
    从论文子网页中获取 Abstract。

    参数:
    - url: 论文子网页的URL字符串。

    返回:
    - Abstract字符串。
    """
    driver.get(url)
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, 'gsc_oci_table'))
    )

    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    abstract_element = soup.find('div', class_='gsh_small')
    abstract = abstract_element.text if abstract_element else ""# 可能为空
    doi = soup.find('a', attrs={'class': 'gsc_oci_title_link'}).get('href')
    values = soup.find_all('div', class_='gsc_oci_value')
    author = values[0].text if len(values) > 0 else ""
    source = values[2].text if len(values) > 0 else ""
    venue = ''
    if '期刊' in html:
        venue = 'Article'
    elif '图书' in html:
        venue = 'Conference paper'
    else:
        venue = ''
    # print(abstract,author,venue)
    return abstract,author,source,venue,doi
# 主逻辑
def main():
    """
    主函数，从Google Scholar网站抓取指定数量的论文信息，并保存到Excel文件中。
    """
    # 定义列名
    columns = ['Year', 'Sources', 'Name', 'Authors', 'First Author', 'Chinese/English', 'Abstract', 'Venues', 'doi']

    # 初始化DataFrame，用于存储论文信息
    papers_df = pd.DataFrame(columns=columns)

    # Loop settings
    cstart = 0  # Starting index of papers
    pagesize = 100  # Number of papers per page

    # 初始化一个空列表来存储每行的数据
    data_rows = []
    i=0
    # 每次爬取100篇，直到爬取所有文章
    while True:
        url = f'{url_base}&cstart={cstart}&pagesize={pagesize}'

        print(url)
        # 使用get_soup函数获取Google Scholar网页的BeautifulSoup对象
        soup = get_soup(url)

        # 找到所有表示论文的行
        rows = soup.find_all('tr', class_='gsc_a_tr')
        print(len(rows))
        try:
            for row in rows:
                # 提取论文标题
                title = row.find('a', class_='gsc_a_at').text
                # 提取论文链接
                paper_url = row.find('a', class_='gsc_a_at')['href']
                print(paper_url)
                # 提取作者信息
                authors = row.find('div', class_='gs_gray').text
                # 提取第一作者
                first_author = authors.split(',')[0] if authors else 'N/A'
                # 提取发表年份
                year_element = row.find('span', class_='gsc_a_h gsc_a_hc gs_ibl')
                year = year_element.text if year_element else ''
                if year == '' or int(year) < 2020:
                    continue  # 忽略2020年之前的论文

                # 提取期刊信息，并去除可能的页码等信息
                journal_info = row.find_all('div', class_='gs_gray')[1].text.split(',')[0] if len(
                    row.find_all('div', 'gs_gray')) > 1 else 'N/A'
                # journal_name = re.sub(r'\s\d+.*$', '', journal_info)
                journal_name = re.sub(r'\s\d+\s*\([^)]*\)', '', journal_info)

                # 判断中文还是英文
                if ',' in authors:
                    Chinese_English = 'English'
                elif '，' in authors:
                    Chinese_English = 'Chinese'
                else:
                    Chinese_English = ''

                time.sleep(1)
                try:
                    abstract,authors,sources,venues,doi = get_abstract('https://scholar.google.com'+paper_url)
                except Exception as e:
                    print(222)
                first_author = authors.split(',')[0] if authors else 'N/A'

                # 输出提示
                print(
                    f"题目：{title}\n作者：{authors}\n一作：{first_author}\n年份：{year}\n期刊：{sources}\n"
                    f"中英文：{Chinese_English}\nAbstract: {abstract}\nVenue: {venues}\ndoi: {doi}\n")

                # 构造论文信息字典
                data = {
                    'Year': year,
                    'Sources': journal_name,
                    'Name': title,
                    'Authors': authors,
                    'First Author': first_author,
                    'Chinese/English': Chinese_English,
                    'Abstract': abstract,
                    'Venues': venues,
                    'doi': doi
                }
                # 将论文信息添加到列表中
                data_rows.append(data)

            cstart += 100
            time.sleep(1)
        except:
            # 所有文献检索完成
            break
    # 将列表转换为DataFrame
    papers_df = pd.DataFrame(data_rows, columns=columns)

    # 获取当前的日期和时间
    current_time = datetime.now()
    # 格式化日期和时间为字符串，用于文件名
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    # 创建文件名，包含当前的日期和时间
    filename = f'changtengfei.csv'
    # 将DataFrame保存到Excel文件中
    papers_df.to_csv(filename, index=False, encoding='utf-8')

    # 输出提示
    print('文件已经生成')


if __name__ == "__main__":
    main()
    driver.quit()  # 关闭浏览器

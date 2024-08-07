from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.chrome.service import Service
import random,re,json
import pandas as pd

#从已有的csv文件中读取论文名字
def read_name(folder):
    df = pd.read_csv(folder,encoding='gbk')
    paper_name = df['Name'].values
    with open('G:\VIStory\project\src\Google_scholar\scholars.txt','w',encoding='utf-8') as file:
        for item in paper_name:
            file.write(str(item) + "\n")

def open_txtfile(folder, file_path):
    paper_id = []
    df = pd.read_csv(folder, encoding='utf-8')
    with open(file_path,'r') as file:
        lines = file.readlines()
    for i in lines:
        if '@' in i and 'Proceedings' not in i:
            paper_id.append(i[i.find('{')+1:i.find(',')])
    print(paper_id,len(paper_id))
    ID = pd.DataFrame({'id':paper_id})
    # df['Id'] = paper_id
    ID.to_csv('xionghui222' + '.csv', index=False, encoding='utf-8')

def extract_href(input_file, output_file, chrome_driver,folder):
    scholars = open(input_file,encoding="utf-8")
    scholars = scholars.readlines()
    browser = webdriver.Chrome(service=Service(executable_path='G:\VIStory\project\src\Google_scholar\chromedriver.exe'))
    print(By.ID)
    url = "https://scholar.google.com"
    browser.get(url)
    # time.sleep(1)
    links = []
    citations = []
    file_out = open(output_file, 'a')
    file_bib_out = open('G://VIStory/project/src/Google_scholar/bib.txt', 'a')
    bibs = []
    # cc = 0
    # st = 64
    failed = []
    pdf_links = {}
    for idx,tt in enumerate(scholars):
        print(idx)
        # cc += 1
        # if not (cc <= 64-18+1 and cc>=44-18+1):
        #     continue
        tt = tt.strip().split('\t')
        tt = tt[-1]
        browser.get(url)
        time.sleep(random.randint(2,10))
        browser.find_element("xpath", '//*[@name="q"]').send_keys(tt)
        time.sleep(3)

        try:
            # el.send_keys(Keys.ENTER)
            browser.find_element("xpath",'//*[@name="btnG"]').click()
            # time.sleep(1)

            browser.find_element("xpath",'//*[@class="gs_or_cit gs_or_btn gs_nph"]').click()
            time.sleep(random.randint(1,3))
            link = browser.find_element("xpath",'//*[@class="gs_citi"]').get_attribute('href')
            print(link)

            # 获取页面源代码
            page_source = browser.page_source
            # 使用正则表达式提取引用次数
            match = re.search(r'被引用次数：(\d+)', page_source)
            if match:
                citation_count = match.group(1)
                citations.append(citation_count)
                print(f"Citation count for {tt}: {citation_count}")
            else:
                citations.append(str(0))
            print(citations,len(citations))
            if link not in links:
                links.append(link)
                file_out.write(link + '\n')

            # 提取页面中所有的链接
            all_pdf_links = browser.find_elements(By.TAG_NAME, 'a')
            time.sleep(random.randint(1, 3))
            # 打印所有链接中包含 '.pdf' 的链接
            for i in all_pdf_links:
                href = i.get_attribute('href')
                if href and 'pdf' in href:
                    print('---pdf----')
                    pdf_links[tt] = href
            print(pdf_links,len(pdf_links))


            browser.get(link)
            text = browser.find_element("xpath",'/html/body/pre')
            text = text.text + '\n'
            file_bib_out.writelines(text)
            bibs.append(text)
            # break

        except:
            print('[*****************************]')
            failed.append(tt)
            continue
        # time.sleep(3)

    print(links)
    print(bibs)
    file_out.close()
    file_bib_out.close()

    print(citations)
    df = pd.read_csv(folder, encoding='gbk')
    df['Citation'] = citations
    df.to_csv('xionghui2' + '.csv', index=False, encoding='utf-8')
    with open('xionghui2_pdf.json', "w") as json_file:
        json.dump(pdf_links, json_file, indent=4)




if __name__ == '__main__':
    input_file = 'G:\VIStory\project\src\Google_scholar\scholars.txt'
    output_file = 'G:\VIStory\project\src\Google_scholar\links.txt'
    folder = "G://VIStory/project/src/xionghui2.csv"
    # pdf_folder = 'G:\VIStory\project\hkust_papers\CMA\zengwei'
    # extract_href(input_file, output_file, None, folder)

    # read_name(folder)

    file_path = 'G://VIStory/project/src/Google_scholar/bib.txt'
    open_txtfile(folder, file_path)

    #测试用什么格式打开csv
    # df = pd.read_csv(folder, encoding='gbk')
    # print(df)


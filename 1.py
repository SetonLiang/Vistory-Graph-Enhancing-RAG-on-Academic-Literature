import aspose.words as aw
from aspose.words import Document
import os,json,re
import pandas as pd

def natural_sort_key(s):
    """Sort string in a way that humans expect."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def find_pdfs(folder_path,filenames,keywords):
    abstracts_len, keywords_len = 0,0
    i = 0
    for root, dirs, files in os.walk(folder_path):
        files.sort(key=natural_sort_key)
        for file in files:
            if file.endswith(".pdf"):
                print(i)
                effect_file = os.path.join(root, file)
                print(effect_file)
                doc = Document(effect_file)
                for para in doc.get_child_nodes(aw.NodeType.PARAGRAPH, True):
                    try:
                        text = para.get_text()
                    except Exception as e:
                        print(f"Error reading paragraph in {effect_file}: {e}")
                        continue
                    # if "Abstract" in text:
                    #     # print(f"Abstract:{text}")
                    #     abstracts.append(text)
                    if "Index Terms" in text:
                        print(f"Keywords:{text}")
                        keywords.append(text)
                    elif "Additional Key Words and Phrases" in text:
                        print(f"Keywords:{text}")
                        keywords.append(text)
                    elif "KEYWORDS" in text or "Keywords" in text:
                        # 分割关键词段落，获取关键词后面的部分
                        parts = text.split("KEYWORDS", 1) if "KEYWORDS" in text else text.split("Keywords", 1)
                        if len(parts) > 1 and parts[1].strip():
                            # 关键词后面有内容
                            keyword = parts[1].strip()
                            print(f"Keywords: {keyword}")
                            keywords.append(keyword)
                        else:
                            # 关键词后面没有内容，获取下一个段落
                            next_para = para.next_sibling
                            if next_para and next_para.node_type == aw.NodeType.PARAGRAPH:
                                try:
                                    next_text = next_para.get_text()
                                    print(f"Keywords: {next_text}")
                                    keywords.append(next_text)
                                except Exception as e:
                                    print(f"Error reading next paragraph in {effect_file}: {e}")
                            else:
                                print(f"No next paragraph found for keywords in {effect_file}")

                filenames.append(effect_file)
                # abstracts_len += 1
                keywords_len += 1
                # if abstracts_len != len(abstracts): abstracts.append(' ')
                if keywords_len != len(keywords): keywords.append(' ')
                i+=1
    return filenames, keywords

def save_json(json_file_path,abstracts, keywords):
    data_dict = [{"Filename": abstract, "Keyword":keyword}for abstract, keyword in zip(abstracts,keywords)]
    with open(json_file_path, "w") as json_file:
        json.dump(data_dict, json_file, indent=4)
def normalize_filename(name):
    """Normalize the filename by replacing spaces with underscores and converting to lower case."""
    return re.sub(r'[\W_]+', '', name.lower())

def json_csv(folder, json_file_path):
    # 读取 CSV 文件
    df = pd.read_csv(folder, encoding='gbk')

    # 添加一个列来存储关键词（如果尚不存在）
    if 'Keyword' not in df.columns:
        df['Keywords'] = ''
    df['NormalizedName'] = df['Name'].apply(lambda x: normalize_filename(x))
    print(df['NormalizedName'])
    # 读取 JSON 文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        contents = json.load(f)

        for content in contents:
            # 获取 JSON 中的文件名和关键词
            filename = content['Filename']
            keyword = content['Keyword']

            # 提取文件名，不包括路径
            paper_name = normalize_filename(os.path.basename(filename).replace('.pdf', ''))
            print(paper_name)
            # 匹配 CSV 中的 Name 列
            match_index = df[df['NormalizedName'] == paper_name].index

            if not match_index.empty:
                # 如果找到匹配行，写入关键词
                df.loc[match_index, 'Keyword'] = keyword
            else:
                print(f"Warning: No match found for {paper_name} in CSV file.")

    # 保存更新后的 CSV 文件
    df.to_csv(folder, index=False, encoding='utf-8')


    # with open('G:\VIStory\project\src\Google_scholar\scholars.txt', 'w', encoding='utf-8') as file:
    #     for item in paper_name:
    #         file.write(str(item) + "\n")


# Example usage:
folder_path = "G:\VIStory\project\hkust_papers\AI/huxuming"
json_file_path = "huxuming.json"
folder = "huxuming.csv"
abstracts = []
keywords = []
filenames = []
text = ''
# filenames, keywords = find_pdfs(folder_path, filenames, keywords)
# print(filenames,keywords)
# save_json(json_file_path, filenames, keywords)

json_csv(folder, json_file_path)


# #用来做测试
# doc = Document("G://VIStory/project/hkust_papers/CMA/zengwei/2.pdf")
# # doc.save("src/output.docx",aw.SaveFormat.DOCX)
# for para in doc.get_child_nodes(aw.NodeType.PARAGRAPH, True):
#     # print(type(para.get_text()))
#     text += str(para.get_text())
#
# # print(text)
# begin = text.find('Abstract—')
# print(begin)
# mid = text.find('Index Terms—')
# # print(text[begin + len('Abstract—'):mid])
# print(text[mid:][len('Index Terms-'):text[mid:].find('.')])

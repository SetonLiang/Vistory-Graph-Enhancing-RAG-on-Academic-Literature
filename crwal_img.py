import os
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
import re
import requests
import base64
import aspose.words as aw
from aspose.words import Document
from PIL import Image
from io import BytesIO
import pandas as pd
import difflib
from pdf2image import convert_from_path


def pdf_to_html(pdf_path: str, html_path: str) -> str:
    """
    将PDF文件转换为HTML文件
    """
    # 打开PDF文件
    doc = fitz.open(pdf_path)
    html_content = "<html><body>"

    # 遍历每一页，将内容转换为HTML
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("html")
        html_content += text

    html_content += "</body></html>"

    # 保存为HTML文件
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return html_path

def del_aspose_elemet(html_tar_file: str, to_tar_file: str):
    """
    去除aspose的信息
    """
    with open(html_tar_file, "r", encoding="utf-8") as html_content:
        soup = BeautifulSoup(html_content, features="lxml")
        # 删除指定的aspose的内容
        for tag in soup.find_all(style=re.compile("-aw-headerfooter-type:")):
            tag.extract()
        word_key_tag = soup.find("p", text=re.compile("Evaluation Only"))
        if word_key_tag:
            word_key_tag.extract()

        with open(to_tar_file, "w", encoding="utf-8") as f:
            f.write(soup.prettify())

def get_same_element_index(ob_string, char):
    return [idx.start() for idx in re.finditer(char, ob_string)]


def extract_images_from_html(html_file_path: str, output_dir: str, filename: str):
    """
    从HTML文件中提取图片并保存到指定目录
    """
    # 创建输出目录，如果不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取HTML文件内容
    with open(html_file_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")
        html_code = soup.prettify()

    img_begins = [m.start() for m in re.finditer('src="data:image', html_code)]

    for index, img_begin in enumerate(img_begins, start=1):
        img_end = html_code.find('"', img_begin + len('src="data:image'))
        img = html_code[img_begin + len('src="data:image/'): img_end]
        head, context = img.split(",", 1)
        img_data = base64.b64decode(context)

        with open(f"{output_dir}/{filename}_{index}.png", "wb") as img_file:
            img_file.write(img_data)

        print(f"保存图像 {index} 成功！")


def create_folders_for_pdfs(directory):
    """
    遍历指定目录中的每个PDF文件，并为每个PDF文件创建一个以该文件名为名字的文件夹。

    :param directory: 要遍历的目录路径
    """
    if not os.path.isdir(directory):
        print(f"{directory} is not a valid directory.")
        return

    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            name_without_extension = os.path.splitext(filename)[0]
            new_folder_path = os.path.join(directory, name_without_extension)
            new_pdf_name = os.path.join(directory, filename)

            try:
                os.makedirs(new_folder_path, exist_ok=True)
                print(f"Folder created: {new_folder_path}")

                new_folder_path2 = os.path.join(new_folder_path, name_without_extension)
                new_html_name = pdf_to_html(new_pdf_name, new_folder_path2 + ".html")

                extract_images_from_html(new_html_name, new_folder_path, name_without_extension )
            except OSError as e:
                print(f"Error creating folder {new_folder_path}: {e}")


def modify_imgname(csv_path, folder_path, threshold=0.6):
    # 读取CSV文件
    df = pd.read_csv(csv_path, encoding='utf-8')
    paper_id = df['Id'].values
    paper_name = df['Name'].values

    # 创建一个名字到Id的映射
    name_to_id = dict(zip(paper_name, paper_id))
    print(name_to_id)

    # 用于跟踪文件名的重复情况
    existing_filenames = set()

    # 遍历文件夹中的文件
    for filename in os.listdir(folder_path):
        # 获取文件的完整路径
        file_path = os.path.join(folder_path, filename)

        # 获取文件名和扩展名
        file_name, file_ext = os.path.splitext(filename)

        # 找到最接近的匹配
        closest_match = difflib.get_close_matches(file_name, paper_name, n=1, cutoff=threshold)

        if closest_match:
            matched_name = closest_match[0]
            base_new_filename = str(name_to_id[matched_name])
            new_filename = base_new_filename + file_ext
            i = 1

            # 如果新文件名已存在，添加索引
            while new_filename in existing_filenames:
                new_filename = f"{base_new_filename}_{i}{file_ext}"
                i += 1

            # 添加新文件名到已存在文件名集合
            existing_filenames.add(new_filename)

            # 新的文件路径
            new_file_path = os.path.join(folder_path, new_filename)

            # 重命名文件
            os.rename(file_path, new_file_path)
            print(f'Renamed {file_path} to {new_file_path}')
        else:
            print(f'No match found for {file_path}')

def normalize_filename(name):
    """Normalize the filename by replacing spaces with underscores and converting to lower case."""
    name = name.replace("-", "").replace("’", "")
    return re.sub(r'[\W_]+', '', name.lower())

def pdfs_to_images(folder,input_dir, output_dir, zoom_x, zoom_y, rotation_angle):
    df = pd.read_csv(folder, encoding='gbk')
    df['normalized_name'] = df['Name'].apply(lambda x: normalize_filename(x))
    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".pdf"):
            print(filename)
            pdf_path = os.path.join(input_dir, filename)
            # Open the PDF file
            pdf = fitz.open(pdf_path)

            # Process only the first page
            page = pdf[0]
            # Set the transformation matrix for scaling and rotation
            trans = fitz.Matrix(zoom_x, zoom_y).prerotate(rotation_angle)
            # Render the page to an image
            pm = page.get_pixmap(matrix=trans, alpha=False)

            normalized_filename = normalize_filename(os.path.splitext(filename)[0])
            # Construct the full image path with the PDF name
            # img_name = f"{os.path.splitext(filename)[0]}.png"
            row = df[df['normalized_name'] == normalized_filename]
            if not row.empty:
                img_name = f"{row['Id'].values[0]}.png"
            else:
                # If no match is found, use the normalized filename as a fallback
                print(222)
                img_name = f"{normalized_filename}.png"
            # print(img_name)
            img_path = os.path.join(output_dir, img_name)
            # Write the image to a PNG file
            pm.save(img_path)

            pdf.close()


def check_duplicates(folder):
    df = pd.read_csv(folder, encoding='gbk')
    # Check for duplicates in the 'Id' column
    duplicate_ids = df[df.duplicated('Id', keep=False)]

    if not duplicate_ids.empty:
        print("Duplicate IDs found:")
        print(duplicate_ids)
    else:
        print("No duplicate IDs found.")


if __name__ == '__main__':

    # 下面是先转html->爬图片
    pdf_file_path = r"G:\VIStory\project\hkust_papers\DSA\chuxiaowen\Energy-aware non-preemptive task scheduling with deadline constraint in dvfs-enabled heterogeneous clusters.pdf"
    html_file_path = "2.html"
    # pdf_to_html(pdf_file_path, html_file_path)
    # output_dir = "G:\VIStory\project\src"
    # process_file_path = r"11.html"
    # del_aspose_elemet(html_file_path, process_file_path)


    #创建文件夹
    directory_path = 'G:\VIStory\project\hkust_papers\CMA\zengwei2'  # 将此路径替换为你的目录路径
    # create_folders_for_pdfs(directory_path)

    # 从HTML中提取图片
    # extract_images_from_html(html_file_path, output_dir)

    csv_path = 'G:\VIStory\project\src\zengwei.csv'
    folder_path = 'G:\VIStory\project\hkust_papers\CMA\zengwei2\Images'
    # modify_imgname(csv_path,folder_path)


    # with open('G://VIStory/project/src/1.html', 'r') as wb_data:  # python打开本地网页文件
    #     Soup = BeautifulSoup(wb_data, 'lxml')  # 建立Soup对象，随后用select函数选取所需部分
    #     html_code = Soup.prettify()


    # 下面是直接pdf转img
    # 将PDF文件转换为图片
    # 先检查是否有重复的ID

    folder = "G:\VIStory\project\hkust_papers\DSA\All\\chuxiaowen.csv"
    input_dir ="G:\VIStory\project\hkust_papers\DSA\\chuxiaowen"
    output_dir = "G:\VIStory\project\hkust_papers\DSA\All\DSA_Images\\chuxiaonwen"
    check_duplicates(folder)
    # pdfs_to_images(folder,input_dir, output_dir, 10, 10, 0)


    #single pdf
    pdf = fitz.open(pdf_file_path)

    # Process only the first page
    page = pdf[0]
    # Set the transformation matrix for scaling and rotation
    trans = fitz.Matrix(10, 10).prerotate(0)
    # Render the page to an image
    pm = page.get_pixmap(matrix=trans, alpha=False)
    pm.save("wang2022energy.png")

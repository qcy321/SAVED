# coding: utf-8
import pandas as pd
import re
from collections import defaultdict
import emoji
import string
from tqdm import tqdm
import time


def pre_clean(text):
    cleaned = f"{text}"
    cleaned = cleaned.replace("（该条评论已经被删除）", "").strip()
    cleaned = cleaned.replace("\n", " ")
    return cleaned


def clean_content(text):
    cleaned = re.split(r'-{10,}.*', text, flags=re.DOTALL)[
        0].strip()  # 去除开发者回复内容
    cleaned = re.sub(r'([!?。，～！？.,;:])\1+', r'\1', cleaned)  # 合并连续重复的标点（如 ！！！ → ！）
    cleaned = emoji.demojize(cleaned, language='zh').replace(":", "'")
    cleaned = cleaned.replace('\t', ' ')
    cleaned = cleaned.replace('-*-', ' ')
    chinese_punctuation = r"，。！？、【】《》；：‘’“”"
    punctuation = re.escape(string.punctuation)  # 转义所有标点符号
    pattern = rf"[^\u4e00-\u9fa5\w\s{punctuation}{chinese_punctuation}]"
    cleaned = re.sub(pattern, " ", cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)  # 去除重复空格
    if cleaned.strip() == " " or cleaned.strip() == "":
        return "内容为空"
    return cleaned


def clean_title(text):
    if not isinstance(text, str):  # 处理 NaN、None、数字等情况
        return "标题为空"
    cleaned = re.sub(r'([!?。，～！？.,;:])\1+', r'\1', text)  # 合并连续重复的标点（如 ！！！ → ！）
    cleaned = emoji.demojize(cleaned, language='zh').replace(":", "'")

    chinese_punctuation = r"，。！？、【】《》；：‘’“”"
    punctuation = re.escape(string.punctuation)  # 转义所有标点符号
    pattern = rf"[^\u4e00-\u9fa5\w\s{punctuation}{chinese_punctuation}]"
    cleaned = re.sub(pattern, " ", cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)  # 去除重复空格
    return cleaned.strip() if cleaned.strip() else "标题为空"


def longest_common_substring(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_length, end_index = 0, 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    end_index = i

    return s1[end_index - max_length:end_index]


def process_excel(input_file, output_file):
    df = pd.read_excel(input_file)

    df['Row'] = df.index
    df['content'] = df['content'].apply(pre_clean)
    df['title'] = df['title'].apply(clean_title)

    grouped = defaultdict(list)
    # 按用户分组
    for _, row in tqdm(df.iterrows()):
        grouped[row.iloc[1]].append((row['Row'], row.iloc[4]))

    for key, items in tqdm(grouped.items()):
        if len(items) > 1:
            items.sort()
            for i in range(len(items) - 1):
                idx1, str1 = items[i]
                idx2, str2 = items[i + 1]
                lcs = longest_common_substring(str1, str2)
                if lcs:
                    df.at[idx2, df.columns[4]] = df.at[idx2, df.columns[4]].replace(lcs, '')

    df['content'] = df['content'].apply(clean_content)
    print("内容清洗完成，写入数据")
    # 保存为txt文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            f.write('-*-'.join(map(str, row[:5])) + '\n')


def format_elapsed_time(start, end):
    elapsed = end - start
    hours = int(elapsed // 3600)
    remaining = elapsed % 3600
    minutes = int(remaining // 60)
    seconds = int(remaining % 60)
    return f"{hours:02d}小时{minutes:02d}分{seconds:02d}秒"


if __name__ == '__main__':
    input_excel = '../data/review/tencent/tencent_review.xlsx'
    output_txt = '../data/review/tencent/tencent_review_cleaned.txt'

    start_time = time.time()
    process_excel(input_excel, output_txt)
    end_time = time.time()

    print(format_elapsed_time(start_time, end_time))

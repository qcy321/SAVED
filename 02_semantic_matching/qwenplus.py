# coding: utf-8
from openai import OpenAI
import time

api_key = 'api-key'
batch_size = 50

client = OpenAI(api_key=api_key, base_url='https://dashscope.aliyuncs.com/compatible-mode/v1')


def batch_process(batch_records, batch_num):
    client = OpenAI(api_key=api_key, base_url='https://dashscope.aliyuncs.com/compatible-mode/v1')

    system_prompt = """
    您是一个专业的App更新分析引擎，请按以下规则处理输入数据：
    1. 读取输入数据
    每条记录格式：record[序号]：更新日志：{update_log}\t\t用户评论：标题-{title}；内容-{content}\n
    2. 语义分析流程
    a) 评论解析
        - 分析用户评论标题及内容的深层语义
        - 识别用户表达的核心诉求/反馈问题
        - 提取具体功能请求或BUG报告关键词
    b) 日志解析：
        - 理解更新日志的技术实现维度
        - 提取功能改进点及问题修复项
    c) 关联性验证：
        - 建立需求关键词与更新要点的映射关系
        - 评估是否满足以下任一条件：
            a) 直接解决用户反馈的问题
            b) 通过功能新增响应潜在需求
        - 不要过度理解日志解决了评论的诉求
        - 仅基于明确证据判断，避免推测
    3. 输出规则
        - 判断结果仅分为[响应]或[未响应]两种，无第三种情况。
        - 输出格式必须严格遵循：
            record1:[响应]
            record2:[未响应]
            ...
        - 不要添加任何解释或额外文本
    """

    user_content = ""
    for i, record in enumerate(batch_records, 1):
        parts = record.strip().split('\t')
        update_log = parts[0].split('-')[3]
        review_parts = parts[1].split('-*-')

        try:
            title = review_parts[3]
            content = review_parts[4].strip()
            # record[序号]：更新日志：{update_log}\t\t用户评论：标题-{title}；内容-{content}\n
            user_content += f"record[{i}]：更新日志：{update_log}\t\t用户评论：标题-{title}；内容-{content}\n"
        except IndexError:
            user_content += f"record{i}:\n[格式错误]\n\n"
    # print(user_content)
    try:
        response = client.chat.completions.create(
            model='qwen-plus',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            stream=False,
            seed=42,
            temperature=0,
            top_p=1.0,
        )

        if response.choices:
            output = response.choices[0].message.content
            results = []
            for line in output.split('\n'):
                if line.strip().startswith('record'):
                    results.append(line.split(':')[1].strip())
                elif '[格式错误]' in line:
                    results.append("格式错误")
            return results
        return ["API错误"] * len(batch_records)

    except Exception as e:
        print(f"批次{batch_num} API错误: {str(e)}")
        return None


def single_process(record):
    client = OpenAI(api_key=api_key, base_url='https://dashscope.aliyuncs.com/compatible-mode/v1')

    system_prompt = """
    您是一个专业的App更新分析引擎，请按以下规则处理输入数据：
    1. 读取输入数据
    记录格式：更新日志：{update_log}\t\t用户评论：标题-{title}；内容-{content}\n
    2. 语义分析流程
    a) 评论解析
        - 分析用户评论标题及内容的深层语义
        - 识别用户表达的核心诉求/反馈问题
        - 提取具体功能请求或BUG报告关键词
    b) 日志解析：
        - 理解更新日志的技术实现维度
        - 提取功能改进点及问题修复项
    c) 关联性验证：
        - 建立需求关键词与更新要点的映射关系
        - 评估是否满足以下任一条件：
            a) 直接解决用户反馈的问题
            b) 通过功能新增响应潜在需求
        - 不要过度理解日志解决了评论的诉求
        - 仅基于明确证据判断，避免推测
    3. 输出规则
    - 判断结果仅分为[响应]或[未响应]两种，无第三种情况。
    - 输出格式必须严格遵循：
        [响应/未响应]
    - 不要添加任何解释或额外文本
    """

    parts = record.strip().split('\t')
    update_log = parts[0].split('-')[3]
    review_parts = parts[1].split('-*-')

    try:
        title = review_parts[3]
        content = review_parts[4].strip()
    except IndexError:
        print(f"异常记录格式: {record}")
        return "格式错误"
    print('开始处理:' + f"更新日志：{update_log}\t\t用户评论：标题-{title}；内容-{content}")
    response = client.chat.completions.create(
        model='qwen-plus',
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"更新日志：{update_log}\t\t用户评论：标题-{title}；内容-{content}"}
        ],
        stream=False,seed=42,
        temperature=0,
        top_p=1.0,
    )
    return response.choices[0].message.content


def format_elapsed_time(start, end):
    elapsed = end - start
    hours = int(elapsed // 3600)
    remaining = elapsed % 3600
    minutes = int(remaining // 60)
    seconds = int(remaining % 60)
    return f"{hours:02d}小时{minutes:02d}分{seconds:02d}秒"


if __name__ == '__main__':
    start_time = time.time()

    with open('../data/pair/tencent/candidate_pair.txt', 'r', encoding='utf-8') as f:
        records = [line.strip() for i, line in enumerate(f) if 0 <= i < 60000]

    total = len(records)
    success_count = 0

    with open('../data/pair/tencent/qw_result.txt', 'w', encoding='utf-8') as f:
        for idx in range(0, len(records), batch_size):
            batch = records[idx:idx + batch_size]
            batch_num = (idx // batch_size) + 1
            print(f"开始处理-批次[{batch_num}]")
            s_time = time.time()
            batch_results = batch_process(batch, batch_num)

            if batch_results and len(batch_results) == len(batch):
                f.write('\n'.join(batch_results) + '\n')
                success_count += len(batch)
                e_time = time.time()
                print(f"批次[{batch_num}]完成 ({len(batch)}条)，耗时{e_time - s_time}秒")
                continue

            print(f"批次-[{batch_num}] 批量处理失败，开始逐条处理")
            for i, record in enumerate(batch, 1):
                retry = 2
                while retry > 0:
                    try:
                        result = single_process(record)
                        f.write(result + '\n')
                        success_count += 1
                        print(f"\t批次-[{batch_num}] 记录-[{idx + i}] 重试-[{3 - retry}]次后成功")
                        break
                    except Exception as e:
                        error_message = getattr(e, 'response', {}).text if hasattr(e, 'response') else str(e)
                        print(f"\t批次-[{batch_num}] 记录-[{idx + i}] API错误: {error_message}")
                        retry -= 1
                        time.sleep(2)
                else:
                    f.write("处理失败\n")
                    print(f"批次-[{batch_num}] 记录-[{idx + i}] 最终失败")
            se_time = time.time()
            print(f"批次[{batch_num}] 最终完成，耗时{se_time - s_time}秒")

    print(f"处理完成，成功率: {success_count}/{total} ({success_count / total:.2%})")

    end_time = time.time()

    print(format_elapsed_time(start_time, end_time))

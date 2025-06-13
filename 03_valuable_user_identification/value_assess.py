import math

import re
import jieba
import jieba.posseg as pseg
from math import exp
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


class UserMetricsAnalyzer:
    def __init__(self, review_path, response_path):
        self.review_path = review_path
        self.response_path = response_path
        self.user_metrics = {}  # 存储指标
        self.LOGICAL_CONNECTORS = {
            '因果': ['因为', '所以', '由于', '因此', '因而', '于是', '导致', '使得'],
            '转折': ['虽然', '但是', '然而', '可是', '不过', '却', '反而', '尽管', '还是'],
            '递进': ['不但', '而且', '甚至', '更', '并且', '此外', '还', '尤其', '特别是'],
            '条件': ['如果', '就', '假如', '那么', '一旦', '只有', '只要'],
            '假设': ['假设', '假定', '倘若', '要是'],
            '并列': ['以及', '还有', '和', '不仅'],
            '选择': ['或者', '要么', '与其', '不如'],
            '让步': ['即使', '也', '即便', '就算'],
            '总结': ['总之', '总而言之', '综述所属'],
            '对比': ['相比之下', '反之', '相反'],
            '目的': ['为了', '以便', '以免'],
            '举例': ['例如', '比如', '譬如']
        }

    def split_to_sentences(self, review):
        sentences = re.findall(r'(.*?[。！？；;!?\n])', review)
        last_part = re.sub(r'[。！？；;!?\n]+$', '', review.split(sentences[-1])[-1]) if sentences else review
        if last_part.strip():
            sentences.append(last_part.strip())
        return sentences

    def calculate_similarity(self, text1, text2):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    def check_syntax_integrity(self, sentence):
        words = pseg.cut(sentence)
        word_list = [(word, flag) for word, flag in words]
        subject_tags = ['n', 'nr', 'ns', 'nt', 'nz', 'r', 'l']  # 名词、代词等可作为主语
        verb_tags = ['v', 'vd', 'vn']  # 动词及其变体
        object_tags = ['n', 'nr', 'ns', 'nt', 'nz']  # 名词可作为宾语
        has_subject = has_verb = has_object = False
        for word, flag in word_list:
            if flag in subject_tags and not has_subject:
                has_subject = True
            elif flag in verb_tags and not has_verb:
                has_verb = True
            elif flag in object_tags and has_verb and not has_object:
                has_object = True
        if has_subject and has_verb and has_object:
            return 1.0  # 主谓宾齐全
        elif has_subject and has_verb:
            return 0.7  # 主谓结构
        elif has_verb:
            return 0.5  # 只有动词的情况（如祈使句）
        else:
            return 0.3  # 句法不完整

    def semantic_integrity(self, review):
        sentences = self.split_to_sentences(review)
        total_score = 0
        sentence_count = len(sentences)
        has_function_desc = has_problem_report = has_suggestion = False
        for sentence in sentences:
            # 句法完整性
            syntax_score = self.check_syntax_integrity(sentence)
            total_score += syntax_score
            # 是否含有功能描述、问题报告或建议
            words = list(jieba.cut(sentence))
            if any(w in ['功能', '版本', '支持', '适配', '增加', '开发'] for w in words):
                has_function_desc = True
            if any(w in ['问题', '不好', '不能', '卡顿', '闪退', '差', '垃圾'] for w in words):
                has_problem_report = True
            if any(w in ['希望', '建议', '改进', '请求', '求', '麻烦'] for w in words):
                has_suggestion = True
        # 基础分数：句法完整性平均值
        base_score = total_score / sentence_count
        # 额外加分：包含完整的功能描述、问题报告或建议
        extra_score = sum([0.1 for cond in [has_function_desc, has_problem_report, has_suggestion] if cond])
        origin_score = base_score + extra_score
        # 针对结构化内容的识别（如数字列表、多个论点等）
        numbered_items = len(re.findall(r'^\s*\d+[\.\、]', review, re.MULTILINE))
        bullet_points = len(re.findall(r'^\s*[•\-\*]', review, re.MULTILINE))
        has_structured_content = numbered_items > 1 or bullet_points > 1
        # 检测是否有详细描述问题或功能的句子
        detailed_descriptions = [s for s in sentences if len(s) > 20 and any(
            w in s for w in ['当', '时候', '情况下', '过程中', '使用时', '出现', '显示'])]
        has_detailed_description = len(detailed_descriptions) > 0
        if has_structured_content and has_detailed_description:
            return min(origin_score * 1.2, 0.98)
        elif has_structured_content:
            return min(origin_score * 1.15, 0.95)
        elif has_detailed_description:
            return min(origin_score * 1.1, 0.95)
        elif len(review) < 30:
            return origin_score * 0.45
        elif len(review) < 20:
            return origin_score * 0.35
        elif len(review) < 10:
            return origin_score * 0.15
        else:
            return origin_score * 0.6

    def logical_coherence(self, review):
        sentences = self.split_to_sentences(review)
        total_score = 0
        for sentence in sentences:
            words = list(jieba.cut(sentence))
            # 是否有逻辑连接词
            has_connector = any(any(w in conn for w in words) for conn in self.LOGICAL_CONNECTORS.values())
            total_score += 0.6 if has_connector else 0.4  # 无连接词但单句独立
        # 是否有问题描述与解决方案/建议的配对
        has_problem = any(
            any(m in s for m in ['问题', '错误', '不能', '无法', '失败', '崩溃', '卡顿', '为什么']) for s in sentences)
        has_solution = any(
            any(m in s for m in ['建议', '希望', '应该', '可以', '需要', '改进', '优化']) for s in sentences)
        # 是否有因果关系的表达
        has_causality = any('因为' in s and ('所以' in s or any(
            '所以' in sentences[i + 1:min(i + 3, len(sentences))] for i in range(len(sentences) - 1))) for s in
                            sentences)
        # 是否有多角度分析
        has_multiple_perspectives = any(
            any(m in s for m in ['一方面', '另一方面', '首先', '其次', '最后', '此外', '从...角度']) for s in sentences)
        if has_problem and has_solution:  # 有问题与解决方案配对
            return min(total_score * 1.2, 0.95)
        elif has_causality:  # 因果关系
            return min(total_score * 1.15, 0.92)
        elif has_multiple_perspectives:  # 多角度分析
            return min(total_score * 1.15, 0.90)
        else:
            return total_score * 0.6

    def normalize_score(self, score):
        normalized = 1 / (1 + exp(-10 * (score - 0.5)))
        return round(normalized * 10, 2)

    def calculate_score_by_ratio(self, value, total_data):
        sorted_data = sorted(total_data)
        data_min, data_max = sorted_data[0], sorted_data[-1]
        if data_min == data_max:
            return 5.0
        if value == data_min:
            return 0.5
        normalized_score = ((value - data_min) / (data_max - data_min)) * 10
        score = np.log(normalized_score + 1) / np.log(11) * 10
        return max(0.5, min(10.0, round(score, 2)))

    def signal_noise_ratio(self):
        print("-" * 20 + '【用户有效评论比例】' + "-" * 20)
        author_dict = defaultdict(list)
        total_count_dict = defaultdict(int)
        valid_count_dict = defaultdict(int)
        with open(self.review_path, "r", encoding="utf-8") as f:
            reviews = [lines.strip() for lines in f.readlines()]

        print("获取用户数据")
        for review in tqdm(reviews):
            pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})-\*-(.*?)-\*-(\d+)-\*-(.*?)-\*-(.*?)-\*-(\d+)'
            match = re.match(pattern, review.strip())
            time, author, rate, title, content, label = match.group(1), match.group(2), match.group(3), match.group(
                4), match.group(5), match.group(6)

            author_dict[author].append(f"{time}-*-{rate}-*-{title}-*-{content}-*-{label}")
            total_count_dict[author] += 1  # 总评论数
            if label == 1:
                valid_count_dict[author] += 1
        print(f"数据解析完成，共【{len(author_dict)}】名用户")

        print("计算用户有效评论比例")
        for author, reviews in author_dict.items():
            total = total_count_dict[author]
            valid = valid_count_dict[author]
            ratio = (valid / total) * 100 if total > 0 else 0.0
            self.user_metrics[author] = self.user_metrics.get(author, {})
            self.user_metrics[author]['x1'] = round(ratio * 0.1, 2)  # 比例基本为100% 直接使用线性映射
        print(f"数据解析完成，存入用户指标")

    def response_query(self, similarity_threshold=0.7):
        print("-" * 20 + '【用户的响应日志数及被响应评论数】' + "-" * 20)
        with open(self.response_path, "r", encoding="utf-8") as f:
            records = [lines.strip().split('\t') for lines in f.readlines()]
        data = [{"app_log": parts[0], "author": parts[1].split('-*-')[1],
                 "combined_text": f"{parts[1].split('-*-')[3]} {parts[1].split('-*-')[4]}"}
                for parts in records]
        author_data = defaultdict(list)
        print("获取用户数据")
        for item in tqdm(data):
            author_data[item['author']].append(item)
        print(f"数据解析完成，共【{len(author_data)}】名用户的评论被响应")

        log_responses_data = []
        review_responses_data = []

        print("计算用户的响应日志数及被响应评论数")
        for author, items in author_data.items():
            unique_logs = set(item['app_log'] for item in items)
            log_responses_data.append(len(unique_logs))
            unique_reviews = []
            for item in items:
                if all(self.calculate_similarity(item['combined_text'], uc['combined_text']) <= similarity_threshold
                       for uc in unique_reviews):
                    unique_reviews.append(item)
            review_responses_data.append(len(unique_reviews))  # 添加到所有数据
        for author, items in tqdm(author_data.items()):
            unique_reviews = []
            for item in items:
                if all(self.calculate_similarity(item['combined_text'], uc['combined_text']) <= similarity_threshold
                       for uc in unique_reviews):
                    unique_reviews.append(item)
            unique_logs = set(item['app_log'] for item in items)
            log_score = self.calculate_score_by_ratio(len(unique_logs), log_responses_data)
            review_score = self.calculate_score_by_ratio(len(unique_reviews), review_responses_data)

            self.user_metrics[author] = self.user_metrics.get(author, {})
            self.user_metrics[author]['x2'] = round(log_score, 2)
            self.user_metrics[author]['x3'] = round(review_score, 2)
        print(f"数据解析完成，存入用户指标")

    def semantics_score(self):
        print("-" * 20 + '【用户请求文本描述的详细程度】' + "-" * 20)
        with open(self.response_path, "r", encoding="utf-8") as f:
            records = [lines.strip() for lines in f.readlines()]
        data = [{"original_info": line, "author": line.strip().split('\t')[1].split('-*-')[1],
                 "content": line.strip().split('\t')[1].split('-*-')[4]}
                for line in records]

        print("解析数据")
        author_data = defaultdict(list)
        for item in tqdm(data):
            author_data[item['author']].append(item)

        print(f"数据解析完成，计算用户请求文本描述的详细程度")
        for author, items in tqdm(author_data.items()):
            total_score = 0
            unique_reviews = []
            for item in items:
                if item['content'] not in unique_reviews:
                    semantic_raw = self.semantic_integrity(item['content'])
                    logical_raw = self.logical_coherence(item['content'])
                    semantic_score = self.normalize_score(semantic_raw)
                    logical_score = self.normalize_score(logical_raw)
                    comprehensive = (semantic_score + logical_score) / 2
                    total_score += comprehensive
                    unique_reviews.append(item['content'])
            author_score = total_score / len(unique_reviews)
            self.user_metrics[author] = self.user_metrics.get(author, {})
            self.user_metrics[author]['x4'] = round(author_score, 2)
        print(f"数据解析完成，存入用户指标")

    def request_uniqueness(self):
        with open(self.response_path, 'r', encoding='utf-8') as f:
            records = [lines.strip().split('\t') for lines in f.readlines()]

        user_requests = defaultdict(list)
        log_unique_users = defaultdict(set)

        for log_part, review_part in records:
            parts = log_part.split('-')
            log_content = parts[4]
            author = review_part.split('-*-')[1]
            user_requests[author].append({'line': f"{log_part}\t{review_part}", 'log_content': log_content})
            log_unique_users[log_content].add(author)

        score = 0
        for author, requests in user_requests.items():
            uniqueness_scores = []
            for request in requests:
                unique_users_count = len(log_unique_users[request['log_content']])
                uniqueness_score = max(0.0, min(10.0, round(10 * math.exp(-0.005 * (unique_users_count - 1)), 2)))
                uniqueness_scores.append(uniqueness_score)

            avg_uniqueness = np.mean(uniqueness_scores) if uniqueness_scores else 0
            self.user_metrics[author] = self.user_metrics.get(author, {})
            self.user_metrics[author]['x5'] = round(avg_uniqueness, 2)

    def normalize(self, values):
        if not values:
            return []
        min_val = min(values)
        max_val = max(values)
        if min_val == max_val:
            return [0.5] * len(values)
        return [(v - min_val) / (max_val - min_val) for v in values]

    def calculate_new_metrics(self, exclude_influence=False, exclude_professionalism=False, exclude_creativity=False):

        # 归一化到[0,1]
        x1_normalized = self.normalize(
            [metrics.get('x1', 0) for metrics in self.user_metrics.values() if 'x1' in metrics])
        x2_normalized = self.normalize(
            [metrics.get('x2', 0) for metrics in self.user_metrics.values() if 'x2' in metrics])
        x3_normalized = self.normalize(
            [metrics.get('x3', 0) for metrics in self.user_metrics.values() if 'x3' in metrics])
        x4_normalized = self.normalize(
            [metrics.get('x4', 0) for metrics in self.user_metrics.values() if 'x4' in metrics])
        x5_normalized = self.normalize(
            [metrics.get('x5', 0) for metrics in self.user_metrics.values() if 'x5' in metrics])

        users_with_complete_data = [user for user, metrics in self.user_metrics.items()
                                    if all(k in metrics for k in ['x1', 'x2', 'x3', 'x4', 'x5'])]

        new_metrics = {}
        for i, user in enumerate(users_with_complete_data):
            metrics_dict = {}
            total_components = []

            if not exclude_influence:
                influence = round((x1_normalized[i] + x2_normalized[i] + x3_normalized[i]) / 3, 3)
                metrics_dict['influence'] = influence
                total_components.append(influence)

            if not exclude_professionalism:
                professionalism = round(x4_normalized[i], 3)
                metrics_dict['professionalism'] = professionalism
                total_components.append(professionalism)

            if not exclude_creativity:
                creativity = round(x5_normalized[i], 3)
                metrics_dict['creativity'] = creativity
                total_components.append(creativity)

            total_score = round(sum(total_components), 3)
            metrics_dict['total'] = total_score

            new_metrics[user] = metrics_dict

        return new_metrics

    def run_all_analyses(self):
        self.signal_noise_ratio()
        self.response_query()
        self.semantics_score()
        self.request_uniqueness()

        results = {}

        results['complete'] = self.calculate_new_metrics()

        results['without_influence'] = self.calculate_new_metrics(exclude_influence=True)

        results['without_professionalism'] = self.calculate_new_metrics(exclude_professionalism=True)

        results['without_creativity'] = self.calculate_new_metrics(exclude_creativity=True)

        return results


if __name__ == "__main__":
    analyzer = UserMetricsAnalyzer(
        review_path="../data/review/tencent/predicted_reviews.txt",
        response_path="../data/pair/tencent/consensus_result.txt"
    )
    all_results = analyzer.run_all_analyses()

    versions = [
        ('complete', '完整版本'),
        ('without_influence', '排除影响力'),
        ('without_professionalism', '排除专业性'),
        ('without_creativity', '排除创造性')
    ]

    for version_key, version_name in versions:
        results = all_results[version_key]
        file_path = f"../data/user/tencent/user_scores_{version_key}.txt"

        sorted_results = sorted(results.items(), key=lambda x: x[1]["total"], reverse=True)

        with open(file_path, "w", encoding="utf-8") as f:
            for user, result in sorted_results:
                score_parts = []
                if 'influence' in result:
                    score_parts.append(f"inf:{result['influence']}")
                if 'professionalism' in result:
                    score_parts.append(f"pro:{result['professionalism']}")
                if 'creativity' in result:
                    score_parts.append(f"cre:{result['creativity']}")

                score_str = f"{user}\t{' '.join(score_parts)}\t总分:{result['total']}"
                f.write(score_str + "\n")

        print(f"{version_name}用户评分计算完成，保存到{file_path}文件")

    print("\n所有版本的用户评分计算完成！")
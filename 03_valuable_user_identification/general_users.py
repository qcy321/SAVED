import random
'''
baseline-1：获取数量相同的随机用户
'''

def extract_general_users(review_path, response_path, output_file):
    all_users = set()
    with open(review_path, 'r', encoding='utf-8') as f:
        reviews = [lines.strip() for lines in f.readlines()]

    for review in reviews:
        all_users.add(review.split('-*-')[1])

    value_users = set()
    with open(response_path, 'r', encoding='utf-8') as f:
        records = [lines.strip() for lines in f.readlines()]

    for record in records:
        v_users = record.split('\t')[1].split('-*-')[1]
        value_users.add(v_users)

    candidate_user = all_users - value_users
    if len(candidate_user) >= len(value_users):
        no_resp_users = set(random.sample(candidate_user, int(len(value_users) * 0.3)))
    else:
        no_resp_users = candidate_user

    with open(output_file, 'w', encoding='utf-8')as f:
        for user in no_resp_users:
            f.write(user+'\n')

    print(f"结果写入 {output_file}")

if __name__ == '__main__':
    review_path = '../data/review/tencent/predicted_reviews.txt'   # 评论路径
    response_path = '../data/pair/tencent/consensus_result.txt'  # 响应路径
    output_file = '../data/user/tencent/general_user.txt'  # 随机选择的用户
    # output_file = '../data/user/tencent/general_user2.txt'  # 随机选择的用户
    # output_file = '../data/user/tencent/general_user3.txt'  # 随机选择的用户
    # output_file = '../data/user/tencent/general_user4.txt'  # 随机选择的用户
    # output_file = '../data/user/tencent/general_user5.txt'  # 随机选择的用户

    extract_general_users(review_path, response_path, output_file)

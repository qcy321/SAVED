from collections import Counter


def extract_active_users(review_path, output_file):
    users = []
    with open(review_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('-*-')
            users.append(parts[1])

    user_review_counts = Counter(users)

    frequent_users = {user: count for user, count in user_review_counts.items() if count > 10}

    with open(output_file, 'w', encoding='utf-8') as file:
        if frequent_users:
            for user, count in frequent_users.items():
                file.write(f"{user}\n")
        else:
            file.write("没有用户发布超过10条评论。\n")

    print(f"结果写入 {output_file}")


if __name__ == '__main__':
    review_path = '../data/review/tencent/predicted_reviews.txt'
    output_file = '../data/user/tencent/active_user.txt'
    extract_active_users(review_path, output_file)

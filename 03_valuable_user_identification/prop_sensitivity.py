import os


def extract_users_by_percentage(filename, p):
    users = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            username = line.split('\t')[0].strip()
            users.append(username)
    total_users = len(users)
    num_to_extract = int(total_users * p / 100)
    return users[:num_to_extract]


def save_users(users, p):
    filename = f"../data/user/tencent/users_{int(p)}.txt"
    with open(filename, 'w', encoding='utf-8') as file:
        for user in users:
            file.write(f"{user}\n")
    return filename


if __name__ == "__main__":
    user_path = "../data/user/tencent/user_scores_complete.txt"

    per = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]

    for p in per:
        extracted_users = extract_users_by_percentage(user_path, p)
        output_file = save_users(extracted_users, p)
        print(f"\n{p}% ({len(extracted_users)} 用户) 已保存至: {output_file}")

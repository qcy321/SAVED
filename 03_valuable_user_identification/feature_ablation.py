import os


def extract_top_30_percent_users(file_path, per):
    users = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 0:
                username = parts[0]
                users.append(username)

    total_users = len(users)
    top_30_percent_count = int(total_users * per / 100)

    top_users = users[:top_30_percent_count]

    print(f"文件: {os.path.basename(file_path)}")
    print(f"总用户数: {total_users}")
    print(f"前30%用户数: {top_30_percent_count}")
    print("-" * 50)

    return top_users


def save_results(all_top_users):
    for version, users in all_top_users.items():
        output_file = f"../data/user/tencent/top_30_per_{version}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, user in enumerate(users, 1):
                f.write(f"{user}\n")
        print(f"已保存: {output_file}")


if __name__ == "__main__":
    files_to_process = [
        ("../data/user/tencent/user_scores_complete.txt", "complete"),
        ("../data/user/tencent/user_scores_without_influence.txt", "without_influence"),
        ("../data/user/tencent/user_scores_without_professionalism.txt", "without_professionalism"),
        ("../data/user/tencent/user_scores_without_creativity.txt", "without_creativity")
    ]
    per = 30.0
    print("提取各版本用户...")
    all_top_users = {}
    for file_path, version_name in files_to_process:
        top_users = extract_top_30_percent_users(file_path, per)
        if top_users:
            all_top_users[version_name] = top_users

    print("保存结果文件...")
    save_results(all_top_users)

    print(f"\n处理完成! 共处理了{len(all_top_users)}个版本的数据")

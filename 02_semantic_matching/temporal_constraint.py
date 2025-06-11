# coding: utf-8

def prepare_pair(review_path, log_path, cand_path):
    with open(review_path, 'r', encoding='utf-8') as f:
        reviews = [line.strip() for line in f.readlines()]

    with open(log_path, 'r', encoding='utf-8') as f:
        logs = [line.strip() for line in f.readlines()]

    review_before_log = []

    for log in logs:
        log_date = log[:10]
        for review in reviews:
            if review.split('-*-')[5] == '1':
                review_date = review.split('-*-')[0][:10]
                if review_date < log_date:
                    record = log + '\t' + review
                    review_before_log.append(record)

    print('review_before_log num:', len(review_before_log))

    with open(cand_path, 'w', encoding='utf-8') as f:
        for item in review_before_log:
            f.write(item + '\n')


if __name__ == '__main__':
    log_path = '../data/log/tencent.txt'
    review_path = "../data/review/tencent/predicted_reviews.txt"
    cand_path = '../data/pair/tencent/candidate_pair.txt'

    prepare_pair(review_path, log_path, cand_path)

import torch
import numpy as np
from transformers import BertTokenizer
import os

from chinese_app_review_classification import ChineseReviewClassifier


def load_model(model_path, model_name="../chinese-roberta-wwm-ext-large"):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = ChineseReviewClassifier(model_name=model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device


def predict_reviews(model, tokenizer, texts, device, max_length=128):
    model.eval()
    batch_size = 16
    all_predictions = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        inputs = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = model.predict(logits).cpu().numpy()
            all_predictions.extend(predictions)

    return np.array(all_predictions).flatten()


def load_test_data(file_path):
    texts = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                if '\t' in line:
                    parts = line.strip().split('\t')
                    review = parts[0].split('-*-')
                    title = review[3]
                    content = review[4]
                    text = title + ' ' + content
                else:
                    review = line.strip().split('-*-')
                    title = review[3]
                    content = review[4]
                    text = title + ' ' + content
                texts.append(text)

    return texts


def save_predictions(predictions, output_file, test_file):
    with open(test_file, 'r', encoding='utf-8') as f:
        original_lines = [line.strip() for line in f if line.strip()]

    with open(output_file, 'w', encoding='utf-8') as f:
        for original_line, pred in zip(original_lines, predictions):
            f.write(f"{original_line}-*-{int(pred)}\n")

    print(f"预测结果已保存至 {output_file}")


def predict_and_save(model_path, test_file, output_file, model_name="../chinese-roberta-wwm-ext-large"):
    print("加载模型...")
    model, device = load_model(model_path, model_name)

    print("加载tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(model_name)

    print(f"从 {test_file} 加载测试数据...")
    texts = load_test_data(test_file)
    print(f"加载了 {len(texts)} 条测试数据")

    print("开始预测...")
    predictions = predict_reviews(model, tokenizer, texts, device)

    print("保存预测结果...")
    save_predictions(predictions, output_file, test_file)



if __name__ == "__main__":
    model_path = "model_output/chinese_review_classifier.pth"
    test_file = "../data/review/tencent/tencent_review_cleaned.txt"
    output_file = "../data/review/tencent/predicted_reviews.txt"
    model_name = "../chinese-roberta-wwm-ext-large"  

    predict_and_save(model_path, test_file, output_file, model_name)
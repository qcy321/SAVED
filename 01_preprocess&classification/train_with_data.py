import torch
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import DataLoader, random_split
import os

from chinese_app_review_classification import (
    ChineseReviewClassifier,
    ChineseReviewDataset,
    train_model,
    evaluate_model,
    predict_reviews
)


def load_data_from_txt(file_path):
    texts = []
    labels = []
    error_lines = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:  
                        
                        review = parts[0].split('-*-')
                        title = review[3]
                        content = review[4]
                        text = title + ' ' + content
                        
                        label = int(parts[-1])  

                        if label not in [0, 1]:
                            error_lines.append((line_num, line.strip(), f"标签无效: {label}"))
                            continue

                        texts.append(text)
                        labels.append(label)
                    else:
                        error_lines.append((line_num, line.strip(), "格式不正确，缺少标签或文本"))
                except Exception as e:
                    error_lines.append((line_num, line.strip(), f"处理出错: {str(e)}"))

    if error_lines:
        print(f"发现 {len(error_lines)} 行数据存在问题:")
        for line_num, content, reason in error_lines[:10]:  
            print(f"行 {line_num}: {content[:50]}... - {reason}")
        if len(error_lines) > 10:
            print(f"... 以及 {len(error_lines) - 10} 个其他错误")

    print(f"成功加载 {len(texts)} 条评论数据")
    return texts, labels



if __name__ == "__main__":
    # 加载数据
    data_file = "app_reviews.txt"
    texts, labels = load_data_from_txt(data_file)

    print(f"总共加载了 {len(texts)} 条评论数据")

    label_counts = np.bincount(labels)
    print(f"标签分布: 标签0: {label_counts[0]}, 标签1: {label_counts[1]}")

    model_name = "../chinese-roberta-wwm-ext-large"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = ChineseReviewClassifier(model_name=model_name)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    model.to(device)

    dataset = ChineseReviewDataset(texts, labels, tokenizer)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")

    batch_size = 16
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    print("开始训练模型...")
    epochs = 5
    trained_model = train_model(model, train_dataloader, val_dataloader, device, epochs=epochs)

    metrics = evaluate_model(trained_model, val_dataloader, device)
    print(f"验证集评估结果: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, "
          f"Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")

    output_dir = "model_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_path = os.path.join(output_dir, "chinese_review_classifier.pth")
    torch.save(trained_model.state_dict(), model_path)
    print(f"模型已保存至: {model_path}")

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import os

# 配置参数
class Config:
    data_path = "./作业数据-waimai_10k.csv"
    model_name = 'bert-base-chinese'  # 使用中文BERT
    num_labels = 3  # 假设有3个分类（0,1,2）
    max_len = 128
    batch_size = 16
    epochs = 3
    learning_rate = 2e-5
    output_model_dir = "../app/model"  # 模型保存目录

# 自定义Dataset
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def load_and_preprocess_data():
    """加载和预处理数据"""
    df = pd.read_csv(Config.data_path)
    
    # 检查数据格式
    print("数据前5行:")
    print(df.head())
    print(f"\n数据形状: {df.shape}")
    print(f"标签分布:\n{df['label'].value_counts()}")
    
    # 确保标签是整数
    df['label'] = df['label'].astype(int)
    
    texts = df['review'].values
    labels = df['label'].values
    
    return texts, labels

def train():
    # 1. 加载和预处理数据
    texts, labels = load_and_preprocess_data()
    
    # 划分训练集和验证集
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # 2. 初始化Tokenizer和模型
    tokenizer = BertTokenizer.from_pretrained(Config.model_name)
    model = BertForSequenceClassification.from_pretrained(
        Config.model_name, 
        num_labels=Config.num_labels
    )

    # 3. 创建DataLoader
    train_dataset = ReviewDataset(train_texts, train_labels, tokenizer, Config.max_len)
    val_dataset = ReviewDataset(val_texts, val_labels, tokenizer, Config.max_len)

    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False)

    # 4. 设置设备、优化器和调度器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=Config.learning_rate)
    total_steps = len(train_loader) * Config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # 5. 训练循环
    for epoch in range(Config.epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # 收集预测结果
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # 验证阶段
        model.eval()
        val_preds = []
        val_labels_list = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                preds = torch.argmax(outputs.logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())

        # 计算指标
        train_acc = accuracy_score(all_labels, all_preds)
        val_acc = accuracy_score(val_labels_list, val_preds)
        avg_train_loss = total_loss / len(train_loader)

        print(f'Epoch {epoch + 1}/{Config.epochs}')
        print(f'Average Training Loss: {avg_train_loss:.4f}')
        print(f'Training Accuracy: {train_acc:.4f}')
        print(f'Validation Accuracy: {val_acc:.4f}')
        print('-' * 50)

    # 6. 保存微调后的模型
    os.makedirs(Config.output_model_dir, exist_ok=True)
    model.save_pretrained(Config.output_model_dir)
    tokenizer.save_pretrained(Config.output_model_dir)
    print(f"模型已保存到: {Config.output_model_dir}")
    
    # 输出最终的分类报告
    print("\n验证集详细报告:")
    print(classification_report(val_labels_list, val_preds))

if __name__ == '__main__':
    train()

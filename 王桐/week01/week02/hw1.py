import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 数据加载和预处理
dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)
max_len = 40

class CharBoWDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self._create_bow_vectors()

    def _create_bow_vectors(self):
        tokenized_texts = []
        for text in self.texts:
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            tokenized += [0] * (self.max_len - len(tokenized))
            tokenized_texts.append(tokenized)

        bow_vectors = []
        for text_indices in tokenized_texts:
            bow_vector = torch.zeros(self.vocab_size)
            for index in text_indices:
                if index != 0:
                    bow_vector[index] += 1
            bow_vectors.append(bow_vector)
        return torch.stack(bow_vectors)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.labels[idx]

# 定义不同的模型架构
class SimpleClassifier1(nn.Module):  # 原始模型：1隐藏层，128节点
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleClassifier1, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class SimpleClassifier2(nn.Module):  # 更宽的网络：1隐藏层，256节点
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleClassifier2, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class DeepClassifier(nn.Module):  # 更深的网络：3隐藏层，每层64节点
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DeepClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.relu(self.fc3(out))
        out = self.fc4(out)
        return out

class VeryWideClassifier(nn.Module):  # 非常宽的网络：1隐藏层，512节点
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VeryWideClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 训练函数
def train_model(model, dataloader, num_epochs=10, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(dataloader)
        losses.append(epoch_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    
    return losses

# 准备数据
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)
output_dim = len(label_to_index)

# 定义不同的模型配置
model_configs = [
    {"name": "1层-128节点", "model_class": SimpleClassifier1, "hidden_dim": 128},
    {"name": "1层-256节点", "model_class": SimpleClassifier2, "hidden_dim": 256},
    {"name": "3层-64节点", "model_class": DeepClassifier, "hidden_dim": 64},
    {"name": "1层-512节点", "model_class": VeryWideClassifier, "hidden_dim": 512}
]

# 训练所有模型并记录loss
results = {}
for config in model_configs:
    print(f"\n训练模型: {config['name']}")
    model = config["model_class"](vocab_size, config["hidden_dim"], output_dim)
    losses = train_model(model, dataloader, num_epochs=10)
    results[config["name"]] = losses


# 打印最终loss对比
print("\n各模型最终Loss对比:")
for name, losses in results.items():
    print(f"{name}: {losses[-1]:.4f}")

# 测试函数
def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    tokenized += [0] * (max_len - len(tokenized))

    bow_vector = torch.zeros(vocab_size)
    for index in tokenized:
        if index != 0:
            bow_vector[index] += 1

    bow_vector = bow_vector.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(bow_vector)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label

# 选择最佳模型进行测试
best_model_name = min(results.items(), key=lambda x: x[1][-1])[0]
best_config = next(config for config in model_configs if config["name"] == best_model_name)
best_model = best_config["model_class"](vocab_size, best_config["hidden_dim"], output_dim)

# 重新训练最佳模型
print(f"\n重新训练最佳模型: {best_model_name}")
train_model(best_model, dataloader, num_epochs=10)

index_to_label = {i: label for label, i in label_to_index.items()}

# 测试
new_text = "帮我导航到北京"
predicted_class = classify_text(new_text, best_model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text(new_text_2, best_model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")

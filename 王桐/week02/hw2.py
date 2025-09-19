import torch
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成sin函数数据
np.random.seed(42)  # 设置随机种子保证可重复性
X_numpy = np.random.rand(1000, 1) * 4 * np.pi - 2 * np.pi  # 生成 [-2π, 2π] 范围内的数据
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(1000, 1)  # sin函数加上一些噪声

X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("Sin函数数据生成完成。")
print(f"数据范围: [{X.min().item():.2f}, {X.max().item():.2f}]")
print("---" * 10)

# 2. 定义多层神经网络
class SinNet(torch.nn.Module):
    def __init__(self, hidden_size=64):
        super(SinNet, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_size),  # 输入层 -> 隐藏层1
            torch.nn.ReLU(),                  # 激活函数
            torch.nn.Linear(hidden_size, hidden_size),  # 隐藏层1 -> 隐藏层2
            torch.nn.ReLU(),                  # 激活函数
            torch.nn.Linear(hidden_size, hidden_size),  # 隐藏层2 -> 隐藏层3
            torch.nn.ReLU(),                  # 激活函数
            torch.nn.Linear(hidden_size, 1)   # 隐藏层3 -> 输出层
        )
    
    def forward(self, x):
        return self.network(x)

# 3. 创建模型、损失函数和优化器
model = SinNet(hidden_size=128)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器

print("模型结构:")
print(model)
print("---" * 10)

# 4. 训练模型
num_epochs = 2000
losses = []

for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X)
    
    # 计算损失
    loss = loss_fn(y_pred, y)
    
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    # 每200个epoch打印一次损失
    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

print("\n训练完成！")
print("---" * 10)

# 5. 可视化结果
# 生成测试数据用于绘制平滑曲线
X_test = torch.linspace(-2*np.pi, 2*np.pi, 1000).reshape(-1, 1).float()

# 使用训练好的模型进行预测
model.eval()
with torch.no_grad():
    y_test_pred = model(X_test)
    y_train_pred = model(X)

# 绘制结果
plt.figure(figsize=(15, 10))

# 子图1：拟合结果
plt.subplot(2, 2, 1)
plt.scatter(X_numpy, y_numpy, label='训练数据', color='blue', alpha=0.6, s=10)
plt.plot(X_test.numpy(), y_test_pred.numpy(), label='神经网络拟合', color='red', linewidth=2)
plt.plot(X_test.numpy(), np.sin(X_test.numpy()), label='真实sin函数', color='green', linestyle='--', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Sin函数拟合结果')
plt.legend()
plt.grid(True)

# 子图2：损失曲线
plt.subplot(2, 2, 2)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('训练损失曲线')
plt.yscale('log')  # 使用对数坐标更好地观察损失变化
plt.grid(True)

# 子图3：预测值与真实值对比
plt.subplot(2, 2, 3)
plt.scatter(y.numpy(), y_train_pred.numpy(), alpha=0.6)
plt.plot([-1.5, 1.5], [-1.5, 1.5], 'r--')  # 对角线
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('预测值 vs 真实值')
plt.grid(True)

# 子图4：残差图
plt.subplot(2, 2, 4)
residuals = y.numpy() - y_train_pred.numpy()
plt.scatter(y_train_pred.numpy(), residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('预测值')
plt.ylabel('残差')
plt.title('残差分析')
plt.grid(True)

plt.tight_layout()
plt.show()

# 计算最终评估指标
final_loss = loss_fn(y_train_pred, y).item()
print(f"最终损失: {final_loss:.6f}")

# 在测试集上评估
X_test_eval = torch.linspace(-np.pi, np.pi, 100).reshape(-1, 1).float()
y_test_true = torch.sin(X_test_eval)
with torch.no_grad():
    y_test_pred_eval = model(X_test_eval)
test_loss = loss_fn(y_test_pred_eval, y_test_true).item()

print(f"测试集损失: {test_loss:.6f}")
print("---" * 10)

# 保存模型（可选）
torch.save(model.state_dict(), 'sin_net.pth')
print("模型已保存为 'sin_net.pth'")

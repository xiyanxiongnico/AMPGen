import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
import pickle

# 加载数据
data = pd.read_csv('/media/zzh/data/AMP/LSTMregrssion/data/train5_65_stpa_mean_representations.csv')

# x as feature y as logMIC
X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)
# 保存标准化器
with open('/media/zzh/data/AMP/LSTMregrssion/LSTM_model/stpascaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
# 转换为张量
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)  # 使用 float 类型以适应 MSELoss

# 数据集划分（保留 20% 作为验证集）
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建数据加载器
train_dataset = TensorDataset(X_train.unsqueeze(1), y_train)
val_dataset = TensorDataset(X_val.unsqueeze(1), y_val)

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# 训练和验证函数
def train_and_validate(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs, params):
    train_losses = []
    val_losses = []
    best_r2 = -float('inf')
    # patience_counter = 0

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)  # 修改为 outputs.squeeze() 以匹配 y_batch 维度
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)  # 修改为 outputs.squeeze() 以匹配 y_batch 维度
                val_loss += loss.item()
                y_true.extend(y_batch.cpu().numpy())
                y_pred.extend(outputs.squeeze().cpu().numpy())
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        
        # 计算R²
        r2 = r2_score(y_true, y_pred)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]}, Val Loss: {val_losses[-1]}, R²: {r2}, Params: {params}')
        
        # 保存最好的模型
        if r2 > best_r2:
            best_r2 = r2
            patience_counter = 0
            torch.save(model.state_dict(), '/media/zzh/data/AMP/LSTMregrssion/LSTM_model/4stpa_best_model_checkpoint.pth')   # path of the best model
            print(f'Checkpoint saved at epoch {epoch+1} with R²: {r2}')
        else:
            patience_counter += 1
        
        # 更新学习率
        scheduler.step(val_loss)

        # 早停策略
        # if patience_counter >= patience:
        #     print(f'Early stopping at epoch {epoch+1}')
        #     break

    return train_losses, val_losses

def evaluate_model(model, val_loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(outputs.squeeze().cpu().numpy())
    
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return mse, r2, y_true, y_pred

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 确定的超参数
    best_params = {
        'hidden_size': 128,
        'num_layers': 2,
        'learning_rate': 0.0001,
        'batch_size': 64,
        'num_epochs': 200,
        'dropout_rate': 0.7,
        'weight_decay': 0.0001,  # 添加 weight_decay 参数以实现 L2 正则化
        # 'patience': 10  # 早停策略的耐心参数
    }

    # 使用最佳参数训练模型
    model = LSTMModel(input_size=X_train.size(1), hidden_size=best_params['hidden_size'], num_layers=best_params['num_layers'], output_size=1, dropout_rate=best_params['dropout_rate']).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)

    train_loader = DataLoader(TensorDataset(X_train.unsqueeze(1), y_train), batch_size=best_params['batch_size'], shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val.unsqueeze(1), y_val), batch_size=best_params['batch_size'], shuffle=False)
    
    train_losses, val_losses = train_and_validate(model, criterion, optimizer, scheduler, train_loader, val_loader, best_params['num_epochs'], best_params)
    
    # 绘制训练和验证损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Train and Validation Loss')
    plt.show()
    
    # 加载验证集上表现最好的模型
    model.load_state_dict(torch.load('/media/zzh/data/AMP/LSTMregrssion/LSTM_model/4stpa_best_model_checkpoint.pth'))

    # 验证最终模型并计算 MSE 和 R²
    mse, r2, y_true, y_pred = evaluate_model(model, val_loader)

    print(f"Final MSE: {mse}")
    print(f"Final R²: {r2}")

    # 绘制预测值和真实值的曲线图
    plt.figure(figsize=(20, 7))

    # 仅显示前500个样本
    n_samples_to_plot = 50
    plt.plot(y_true[:n_samples_to_plot], label='True Values', alpha=0.7)
    plt.plot(y_pred[:n_samples_to_plot], label='Predicted Values', alpha=0.7)

    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.title('True vs Predicted Values')
    plt.show()
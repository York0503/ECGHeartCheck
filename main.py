import wfdb
import numpy as np
import os
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 定義資料集類別
class ECGDataset(Dataset):
    def __init__(self, signals, labels):
        self.signals = signals
        self.labels = labels

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]
        return torch.tensor(signal, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# 定義模型類別
class ECGClassifier(nn.Module):
    def __init__(self, input_size=500, num_channels=2):
        super(ECGClassifier, self).__init__()
        self.conv1 = nn.Conv1d(num_channels, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        # 計算特徵大小
        with torch.no_grad():
            x = torch.randn(1, num_channels, input_size)
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            self.feature_size = x.view(1, -1).size(1)

        self.fc1 = nn.Linear(self.feature_size, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 將形狀轉換為 (batch_size, channels, length)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# 訓練模型
def train_model(model, train_loader, val_loader, num_epochs=20):
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # 訓練階段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total

        # 驗證階段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for signals, labels in val_loader:
                signals, labels = signals.to(device), labels.to(device)
                outputs = model(signals)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total

        # 打印訓練信息
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        print('-' * 60)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_ecg_model.pth')


# 讀取 ECG 數據並將其整理
def load_ecg_data_from_folder(root_folder, signal_length=500):
    signals = []
    labels = []

    # 遍歷資料夾下所有的 Person_x 資料夾
    for person_folder in os.listdir(root_folder):
        person_folder_path = os.path.join(root_folder, person_folder)
        if not os.path.isdir(person_folder_path):
            continue

        # 遍歷每個人的 rec_x 目錄
        for record_folder in os.listdir(person_folder_path):
            record_folder_path = os.path.join(person_folder_path, record_folder)
            if not os.path.isdir(record_folder_path):
                continue

            # 確保 rec_x 資料夾中有 .dat 和 .atr 文件
            dat_file = os.path.join(record_folder_path, f"{record_folder}.dat")
            atr_file = os.path.join(record_folder_path, f"{record_folder}.atr")

            if os.path.exists(dat_file) and os.path.exists(atr_file):
                try:
                    # 讀取 ECG 數據和標註
                    record = wfdb.rdrecord(dat_file)
                    annotation = wfdb.rdann(atr_file, 'atr')
                    # 取得心電圖信號與標註
                    ecg_signals = record.p_signal
                    ecg_labels = annotation.symbol

                    # 假設標註 "N" 表示正常心跳，"V" 表示異常，將其轉換為數字標籤
                    label_map = {'N': 0, 'V': 1}
                    numeric_labels = [label_map.get(label, 0) for label in ecg_labels]  # 轉換標註為數字

                    # 這裡進行信號的分割，每次取 500 步長的信號，並對應標註
                    for i in range(0, len(ecg_signals), signal_length):
                        signal_segment = ecg_signals[i:i+signal_length]
                        if len(signal_segment) < signal_length:
                            continue  # 忽略不足500步的片段

                        # 將標註映射到對應的信號段
                        label_index = i // signal_length  # 計算這段信號的標註
                        label = numeric_labels[min(label_index, len(numeric_labels) - 1)]  # 確保索引不超出範圍

                        signals.append(signal_segment)
                        labels.append(label)
                except Exception as e:
                    print(f"Error loading {record_folder}: {e}")
                    continue

    # 檢查是否有數據
    if not signals or not labels:
        raise ValueError("No valid ECG data found. Please check the file paths or annotations.")

    signals = np.array(signals)
    labels = np.array(labels)

    print(f"Signals shape: {signals.shape}")
    print(f"Labels shape: {labels.shape}")

    return signals, labels


# 數據準備
root_folder = os.getcwd() + '\data\ecg-id-database-1.0.0'  # 資料夾路徑
if os.path.exists(root_folder) != True:
    print(f"路徑 {root_folder} 不存在，請檢查路徑是否正確。")
  
else:
    try:
        signals, labels = load_ecg_data_from_folder(root_folder)
    except ValueError as e:
        print(e)
        exit(1)

    # 檢查信號和標註數據是否一致
    if len(signals) != len(labels):
        print(f"Error: Signals and labels have different lengths. Signals: {len(signals)}, Labels: {len(labels)}")
        exit(1)

    # 進行訓練與驗證資料集的分割
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(signals, labels, test_size=0.2, random_state=42)

    # 創建數據加載器
    train_dataset = ECGDataset(X_train, y_train)
    val_dataset = ECGDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 設置設備參數
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGClassifier(input_size=X_train.shape[1], num_channels=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    # 開始訓練
    train_model(model, train_loader, val_loader)

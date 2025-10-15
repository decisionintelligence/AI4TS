import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ========== RNN ==========
class RNNTSCClassifier(nn.Module):
    """RNN 时间序列分类器"""
    def __init__(self, input_length, num_classes, hidden_size=128, num_layers=2, dropout_rate=0.5):
        super(RNNTSCClassifier, self).__init__()

        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )

        self._to_linear = None
        self._get_rnn_output((1, input_length))

        self.classifier = nn.Sequential(
            nn.Linear(self._to_linear, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def _get_rnn_output(self, shape):
        with torch.no_grad():
            x = torch.zeros(1, *shape)  # (1,1,seq_len)
            x = x.transpose(1, 2)       # (1, seq_len, 1)
            out, _ = self.rnn(x)
            self._to_linear = out[:, -1, :].shape[1]

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, seq_len, 1)
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return self.classifier(out)


# ========== LSTM ==========
class LSTMTSCClassifier(nn.Module):
    """LSTM 时间序列分类器"""
    def __init__(self, input_length, num_classes, hidden_size=128, num_layers=2, dropout_rate=0.5):
        super(LSTMTSCClassifier, self).__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )

        self._to_linear = None
        self._get_rnn_output((1, input_length))

        self.classifier = nn.Sequential(
            nn.Linear(self._to_linear, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def _get_rnn_output(self, shape):
        with torch.no_grad():
            x = torch.zeros(1, *shape)
            x = x.transpose(1, 2)
            out, (hn, cn) = self.lstm(x)
            self._to_linear = out[:, -1, :].shape[1]

    def forward(self, x):
        x = x.transpose(1, 2)
        out, (hn, cn) = self.lstm(x)
        out = out[:, -1, :]
        return self.classifier(out)


# ========== GRU ==========
class GRUTSCClassifier(nn.Module):
    """GRU 时间序列分类器"""
    def __init__(self, input_length, num_classes, hidden_size=128, num_layers=2, dropout_rate=0.5):
        super(GRUTSCClassifier, self).__init__()

        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )

        self._to_linear = None
        self._get_rnn_output((1, input_length))

        self.classifier = nn.Sequential(
            nn.Linear(self._to_linear, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def _get_rnn_output(self, shape):
        with torch.no_grad():
            x = torch.zeros(1, *shape)
            x = x.transpose(1, 2)
            out, hn = self.gru(x)
            self._to_linear = out[:, -1, :].shape[1]

    def forward(self, x):
        x = x.transpose(1, 2)
        out, hn = self.gru(x)
        out = out[:, -1, :]
        return self.classifier(out)
    
def load_UCR(dataset):
    """加载UCR数据集的标准函数"""
    data_path = "./data"
    # data_path = "chap5_classification/data"
    train_file = os.path.join(data_path, dataset, dataset + "_TRAIN.tsv")
    test_file = os.path.join(data_path, dataset, dataset + "_TEST.tsv")
    
    print(f"正在读取文件: {train_file}")
    print(f"正在读取文件: {test_file}")
    
    # 检查文件是否存在
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"训练文件不存在: {train_file}")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"测试文件不存在: {test_file}")
    
    train_df = pd.read_csv(train_file, sep='\t', header=None)
    test_df = pd.read_csv(test_file, sep='\t', header=None)
    train_array = np.array(train_df)
    test_array = np.array(test_df)

    print(f"原始训练数据形状: {train_array.shape}")
    print(f"原始测试数据形状: {test_array.shape}")

    # Move the labels to {0, ..., L-1}
    labels = np.unique(train_array[:, 0])
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i

    train = train_array[:, 1:].astype(np.float64)
    train_labels = np.vectorize(transform.get)(train_array[:, 0])
    test = test_array[:, 1:].astype(np.float64)
    test_labels = np.vectorize(transform.get)(test_array[:, 0])

    print(f"处理后训练数据形状: {train.shape}")
    print(f"处理后测试数据形状: {test.shape}")
    print(f"类别数量: {len(labels)}")
    print(f"类别映射: {transform}")
    print(f"训练标签分布: {np.unique(train_labels, return_counts=True)}")
    print(f"测试标签分布: {np.unique(test_labels, return_counts=True)}")

    # Normalization for non-normalized datasets
    non_normalized_datasets = [
        'AllGestureWiimoteX', 'AllGestureWiimoteY', 'AllGestureWiimoteZ',
        'BME', 'Chinatown', 'Crop', 'EOGHorizontalSignal', 'EOGVerticalSignal',
        'Fungi', 'GestureMidAirD1', 'GestureMidAirD2', 'GestureMidAirD3',
        'GesturePebbleZ1', 'GesturePebbleZ2', 'GunPointAgeSpan',
        'GunPointMaleVersusFemale', 'GunPointOldVersusYoung', 'HouseTwenty',
        'InsectEPGRegularTrain', 'InsectEPGSmallTrain', 'MelbournePedestrian',
        'PickupGestureWiimoteZ', 'PigAirwayPressure', 'PigArtPressure',
        'PigCVP', 'PLAID', 'PowerCons', 'Rock', 'SemgHandGenderCh2',
        'SemgHandMovementCh2', 'SemgHandSubjectCh2', 'ShakeGestureWiimoteZ',
        'SmoothSubspace', 'UMD'
    ]
    
    if dataset not in non_normalized_datasets:
        # 标准化数据
        mean = np.nanmean(train)
        std = np.nanstd(train)
        train = (train - mean) / std
        test = (test - mean) / std
        print(f"数据已标准化: mean={mean:.4f}, std={std:.4f}")
    
    # 添加通道维度 [samples, timesteps] -> [samples, timesteps, 1]
    train = train[..., np.newaxis]
    test = test[..., np.newaxis]
    
    return train, train_labels, test, test_labels

def prepare_data(train_data, train_labels, test_data, test_labels, val_ratio=0.2):
    """准备PyTorch数据，并划分验证集"""
    # 转换为PyTorch张量
    # 从 [samples, timesteps, 1] 转换为 [samples, 1, timesteps]
    X_train_tensor = torch.FloatTensor(train_data).transpose(1, 2)  # (batch, 1, seq_len)
    y_train_tensor = torch.LongTensor(train_labels)
    X_test_tensor = torch.FloatTensor(test_data).transpose(1, 2)
    y_test_tensor = torch.LongTensor(test_labels)
    
    print(f"训练张量形状: {X_train_tensor.shape}")
    print(f"测试张量形状: {X_test_tensor.shape}")
    
    # 划分训练集和验证集
    dataset_size = len(X_train_tensor)
    val_size = int(dataset_size * val_ratio)
    train_size = dataset_size - val_size
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    
    print(f"训练集大小: {train_size}, 验证集大小: {val_size}")
    
    return train_subset, val_subset, X_test_tensor, y_test_tensor

def train_model_with_validation(model, train_loader, val_loader, criterion, optimizer, device, epochs=100, patience=15):
    """带验证集的模型训练"""
    model.train()
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    # 保存最佳模型状态
    best_model_state = None
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        epoch_train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_train_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        epoch_val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                epoch_val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)
        
        # 早停和模型保存逻辑
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            # 保存最佳模型状态
            best_model_state = model.state_dict().copy()
            print(f'*** 新的最佳模型: 验证准确率 {val_acc:.2f}% ***')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'训练损失: {avg_train_loss:.4f}, 训练准确率: {train_acc:.2f}%, '
                  f'验证损失: {avg_val_loss:.4f}, 验证准确率: {val_acc:.2f}%, '
                  f'早停计数: {patience_counter}/{patience}')
        
        # 早停检查
        if patience_counter >= patience:
            print(f'\n早停触发! 在 epoch {epoch+1} 停止训练')
            print(f'最佳模型在 epoch {best_epoch}, 验证准确率: {best_val_acc:.2f}%')
            break
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'\n已加载最佳模型 (epoch {best_epoch}, 验证准确率: {best_val_acc:.2f}%)')
    
    return train_losses, val_losses, val_accuracies, best_val_acc

def evaluate_model(model, test_loader, device, num_classes):
    """评估模型在测试集上的表现"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probabilities = []
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()
            
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    avg_test_loss = test_loss / len(test_loader)
    
    print(f"\n测试集结果:")
    print(f"测试损失: {avg_test_loss:.4f}")
    print(f"测试准确率: {accuracy:.4f}")
    print("\n详细分类报告:")
    print(classification_report(all_labels, all_preds, 
                               target_names=[f'Class_{i}' for i in range(num_classes)]))
    
    return accuracy, all_preds, all_labels, avg_test_loss, all_probabilities

def main():
    # 参数设置
    dataset_name = "ACSF1"  # 可以更改为其他UCR数据集名称
    batch_size = 32
    learning_rate = 0.001
    epochs = 200
    patience = 20  # 早停耐心值
    val_ratio = 0.2  # 验证集比例
    model_type = "rnn"  # 可选: "rnn", "lstm", "gru"
    hidden_size = 128  # RNN隐藏层大小
    num_layers = 2  # RNN层数
    dropout_rate = 0.3  # Dropout率
    
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据
    print("正在加载UCR数据集...")
    try:
        train_data, train_labels, test_data, test_labels = load_UCR(dataset_name)
    except Exception as e:
        print(f"加载数据失败: {e}")
        print("请检查数据文件路径和格式")
        return
    
    # 数据预处理和验证集划分
    train_subset, val_subset, X_test_tensor, y_test_tensor = prepare_data(
        train_data, train_labels, test_data, test_labels, val_ratio
    )
    
    # 检查数据有效性
    if len(train_subset) == 0 or len(val_subset) == 0:
        print("错误: 数据为空，请检查数据文件")
        return
    
    # 创建数据加载器
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    input_length = X_test_tensor.shape[2]  # 时间序列长度
    num_classes = len(np.unique(train_labels))
    
    # 根据用户选择创建模型
    if model_type == "rnn":
        model = RNNTSCClassifier(input_length, num_classes, hidden_size, num_layers, dropout_rate).to(device)
        print("使用RNN模型")
    elif model_type == "lstm":
        model = LSTMTSCClassifier(input_length, num_classes, hidden_size, num_layers, dropout_rate).to(device)
        print("使用LSTM模型")
    elif model_type == "gru":
        model = GRUTSCClassifier(input_length, num_classes, hidden_size, num_layers, dropout_rate).to(device)
        print("使用GRU模型")
    else:
        print(f"未知模型类型: {model_type}，默认使用RNN")
        model = RNNTSCClassifier(input_length, num_classes, hidden_size, num_layers, dropout_rate).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5, verbose=True)
    
    print(f"\n模型参数:")
    print(f"输入长度: {input_length}")
    print(f"类别数量: {num_classes}")
    print(f"批量大小: {batch_size}")
    print(f"学习率: {learning_rate}")
    print(f"训练轮数: {epochs}")
    print(f"早停耐心: {patience}")
    print(f"验证集比例: {val_ratio}")
    print(f"Dropout率: {dropout_rate}")
    print(f"模型类型: {model_type}")
    print(f"隐藏层大小: {hidden_size}")
    print(f"RNN层数: {num_layers}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练模型（带验证和早停）
    print("\n开始训练...")
    train_losses, val_losses, val_accuracies, best_val_acc = train_model_with_validation(
        model, train_loader, val_loader, criterion, optimizer, device, epochs, patience
    )
    
    # 在测试集上评估最佳模型
    print("\n在测试集上评估最佳模型...")
    test_accuracy, predictions, true_labels, test_loss, probabilities = evaluate_model(
        model, test_loader, device, num_classes
    )
    
    # 保存最佳模型
    os.makedirs("models", exist_ok=True)
    model_save_path = f"models/{dataset_name}_{model_type}_classifier_best.pth"
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_accuracy': test_accuracy,
        'best_val_accuracy': best_val_acc,
        'input_length': input_length,
        'num_classes': num_classes,
        'dataset_name': dataset_name,
        'model_type': model_type,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }, model_save_path)
    
    print(f"\n最佳模型已保存到: {model_save_path}")
    print(f"验证集最佳准确率: {best_val_acc:.2f}%")
    print(f"测试集最终准确率: {test_accuracy:.4f}")
    
    return test_accuracy, train_losses, val_losses, val_accuracies

if __name__ == "__main__":
    test_accuracy, train_losses, val_losses, val_accuracies = main()
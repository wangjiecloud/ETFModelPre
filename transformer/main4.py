# ===== 基础库导入 =====
import torch  # PyTorch深度学习框架 示例：用于构建神经网络
import torch.nn as nn  # 神经网络模块 示例：定义全连接层nn.Linear
import torch.optim as optim  # 优化器模块 示例：使用Adam优化器
from torch.utils.data import Dataset, DataLoader  # 数据集工具 示例：创建自定义数据集
import pandas as pd  # 数据处理库 示例：处理时间序列数据
import re  # 正则表达式 示例：文本清洗（虽然代码中未直接使用）
from sklearn.metrics import accuracy_score, classification_report  # 评估指标 示例：计算准确率
import joblib  # 模型保存 示例：保存特征工程的scaler
import os
from featureBuilder import *


# ===== 1. 模型架构定义 =====
class NewsEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载中文BERT模型（示例：处理中文财经新闻）
        self.bert = BertModel.from_pretrained('bert-base-chinese', cache_dir=CACHE_DIR)
        self.dropout = nn.Dropout(0.1)  # 随机失活层（示例：防止过拟合）
        self.fc = nn.Linear(768, 64)  # 全连接层（示例：将768维BERT输出压缩到64维）

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # BERT前向传播（示例：input_ids是分词后的数字ID）
        outputs = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True  # Ensure dictionary output
        )
        # Use pooler_output instead of trying to unpack
        pooled = outputs.pooler_output
        return self.fc(self.dropout(pooled))  # 输出压缩后的特征

def generate_news(prev_price, curr_price):
    """生成模拟新闻（示例：根据价格变化选择新闻模板）"""
    # 新闻模板库
    templates = [
        "美联储宣布{action}基准利率{value}个基点，市场预期{expectation}",
        "中东{event}导致油价波动，黄金{trend}情绪升温",
        "最新{data}数据显示通胀{direction}，影响黄金投资决策"
    ]
    # 动态参数生成逻辑
    action = "降息" if np.random.rand() < 0.6 else "维持"
    event = "冲突加剧" if curr_price > prev_price else "局势缓和"
    return np.random.choice(templates).format(
        action=action, 
        value=np.random.randint(0, 50),      # 随机生成利率变动值
        expectation="乐观" if curr_price > prev_price else "谨慎",  # 根据趋势生成预期
        event=event,  # 根据地缘政治状态生成事件描述
        trend="避险" if curr_price > prev_price else "抛售",  # 生成市场情绪关键词
        data="CPI" if np.random.rand() < 0.7 else "PPI",  # 随机选择经济指标
        direction="上升" if np.random.rand() < 0.6 else "下降"  # 随机生成通胀方向
    )

class TimeSeriesProcessor(nn.Module):
    def __init__(self, input_dim=5):
        super().__init__()
        # LSTM网络（示例：输入5个技术指标）
        self.lstm = nn.LSTM(
            input_size=input_dim,  # 输入特征维度（示例：收盘价、MA5等）
            hidden_size=32,       # 隐层维度
            num_layers=2,          # 堆叠层数
            batch_first=True,     # 输入格式为(batch, seq, feature)
            dropout=0.2            # 层间失活率
        )
        # 注意力机制（示例：关注重要时间点）
        self.attention = nn.MultiheadAttention(
            embed_dim=32,   # 输入维度需与LSTM隐层一致
            num_heads=4,    # 多头注意力头数
            batch_first=True
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # LSTM处理时序数据（示例：shape(batch, seq_len, 32)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)  # 自注意力计算
        return attn_out[:, -1, :]  # 取最后一个时间步输出（示例：最新时间点的特征）

# ===== 修改后的模型架构 =====
class MultiTaskFusionPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        # 共享的特征提取部分保持不变
        self.text_encoder = NewsEncoder()
        self.ts_processor = TimeSeriesProcessor(input_dim=5)
        
        # 多任务输出头
        self.regression_head = nn.Sequential(
            nn.Linear(64 + 32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # 预测收益率（回归任务）
        )
        
        self.classification_head = nn.Sequential(
            nn.Linear(64 + 32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid()  # 预测涨跌概率（分类任务）
        )

    def forward(self, text_inputs, ts_inputs):
        text_feats = self.text_encoder(**text_inputs)
        ts_feats = self.ts_processor(ts_inputs)
        combined = torch.cat([text_feats, ts_feats], dim=1)
        
        # 同时输出两个任务的结果
        return {
            'regression': self.regression_head(combined),
            'classification': self.classification_head(combined)
        }

# ===== 修改后的数据生成 =====
def generate_synthetic_data(days=200, funds=2):
    """生成包含收益率标签的数据"""
    np.random.seed(42)
    data = []
    fund_names = [f"黄金ETF_{i+1}" for i in range(funds)]
    
    for fund in fund_names:
        price_history = np.cumsum(np.random.randn(days)) + 100
        for i in range(20, days):
            news_text = generate_news(price_history[i-1], price_history[i])
            
            # 计算实际收益率（回归目标）
            return_rate = (price_history[i] - price_history[i-1]) / price_history[i-1]
            
            # 生成分类标签（1表示上涨）
            trend_label = 1 if return_rate > 0 else 0
            
            data.append({
                'fund': fund,
                'date': f'2023-01-{i:02d}',
                'news': news_text,
                'price_history': price_history[max(0, i-20):i],
                'return_rate': return_rate,  # 新增回归目标
                'trend_label': trend_label   # 原分类目标
            })
    return data

# ===== 修改后的数据集类 =====
class GoldETFDataSet(Dataset):
    def __getitem__(self, idx):
        sample = self.data[idx]
        text_inputs = {k: v.squeeze(0) for k, v in sample['text'].items()}
        return (
            text_inputs,
            sample['ts'],
            torch.tensor(sample['return_rate'], dtype=torch.float32),  # 回归目标
            torch.tensor(sample['trend_label'], dtype=torch.float32)    # 分类目标
        )

# ===== 修改后的训练函数 =====
def train_multi_task_model(model, dataloader, epochs=10, lr=3e-5, alpha=0.7):
    """多任务训练函数"""
    # 定义双损失函数
    regression_criterion = nn.MSELoss()      # 回归任务用MSE
    classification_criterion = nn.BCELoss()  # 分类任务用BCE
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0.0
        reg_loss_total = 0.0
        cls_loss_total = 0.0
        
        for text_inputs, ts_inputs, return_targets, trend_targets in dataloader:
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(text_inputs, ts_inputs)
            
            # 计算双损失
            reg_loss = regression_criterion(outputs['regression'], return_targets.unsqueeze(1))
            cls_loss = classification_criterion(outputs['classification'], trend_targets.unsqueeze(1))
            
            # 加权组合损失
            total_loss = alpha * reg_loss + (1 - alpha) * cls_loss
            
            # 反向传播
            total_loss.backward()
            optimizer.step()
            
            # 统计指标
            total_loss += total_loss.item()
            reg_loss_total += reg_loss.item()
            cls_loss_total += cls_loss.item()
        
        # 打印训练信息
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Total Loss: {total_loss/len(dataloader):.4f} | "
              f"Reg Loss: {reg_loss_total/len(dataloader):.4f} | "
              f"Cls Loss: {cls_loss_total/len(dataloader):.4f}")

    return model

# ===== 修改后的评估函数 =====
def evaluate_multi_task(model, dataloader):
    model.eval()
    return_true, return_pred = [], []
    trend_true, trend_pred = [], []
    
    with torch.no_grad():
        for text_inputs, ts_inputs, return_t, trend_t in dataloader:
            outputs = model(text_inputs, ts_inputs)
            
            # 收集回归结果
            return_true.extend(return_t.numpy().flatten())
            return_pred.extend(outputs['regression'].numpy().flatten())
            
            # 收集分类结果
            trend_true.extend(trend_t.numpy().flatten())
            trend_pred.extend((outputs['classification'] > 0.5).float().numpy().flatten())
    
    # 回归评估指标
    mse = np.mean((np.array(return_true) - np.array(return_pred))**2)
    mae = np.mean(np.abs(np.array(return_true) - np.array(return_pred)))
    
    # 分类评估指标
    accuracy = accuracy_score(trend_true, trend_pred)
    report = classification_report(trend_true, trend_pred)
    
    print("=== 回归任务评估 ===")
    print(f"MSE: {mse:.4f} | MAE: {mae:.4f}")
    print("\n=== 分类任务评估 ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(report)

# ===== 修改后的训练流程 =====
def train_and_save_multi_task():
    synthetic_data = generate_synthetic_data(days=200)
    feature_builder = FeatureBuilder()
    
    processed_data = []
    for sample in synthetic_data:
        features = feature_builder.build_sample(
            sample['news'], 
            sample['price_history']
        )
        processed_data.append({
            'text': features['text'],
            'ts': features['ts'],
            'return_rate': sample['return_rate'],  # 添加回归目标
            'trend_label': sample['trend_label']   # 分类目标
        })
    
    # 数据集划分
    split_idx = int(0.8 * len(processed_data))
    train_data = processed_data[:split_idx]
    test_data = processed_data[split_idx:]
    
    # 创建数据加载器
    train_dataset = GoldETFDataSet(train_data)
    test_dataset = GoldETFDataSet(test_data)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # 初始化并训练多任务模型
    model = MultiTaskFusionPredictor()
    trained_model = train_multi_task_model(model, train_loader)
    
    # 评估
    print("\n多任务模型评估结果：")
    evaluate_multi_task(trained_model, test_loader)
    
    # 保存模型（需要同时保存特征处理器）
    torch.save(trained_model.state_dict(), "multi_task_model.pth")
    joblib.dump(feature_builder.scaler, "multi_task_scaler.pkl")

# ===== 修改后的验证函数 =====
def validate_multi_task_model():
    # 初始化组件
    feature_builder = FeatureBuilder()
    feature_builder.scaler = joblib.load("multi_task_scaler.pkl")
    
    # 加载模型
    model = MultiTaskFusionPredictor()
    model.load_state_dict(torch.load("multi_task_model.pth"))
    model.eval()
    
    # 生成测试数据
    test_samples = generate_synthetic_data(days=50)
    processed_data = []
    
    for sample in test_samples:
        features = feature_builder.build_sample(
            sample['news'],
            sample['price_history']
        )
        processed_data.append({
            'text': features['text'],
            'ts': features['ts'],
            'return_rate': sample['return_rate'],
            'trend_label': sample['trend_label']
        })
    
    # 创建数据加载器
    test_dataset = GoldETFDataSet(processed_data)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # 评估
    print("\n多任务模型验证结果：")
    evaluate_multi_task(model, test_loader)
    
    # 单条预测示例
    test_news = "美联储宣布降息25个基点，市场避险情绪升温"
    test_price = np.array([100, 102, 101, 103, 104])
    features = feature_builder.build_sample(test_news, test_price)
    
    input_text = {k: v.unsqueeze(0) for k, v in features['text'].items()}
    input_ts = features['ts'].unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_text, input_ts)
    
    print(f"\n预测结果：")
    print(f"预测收益率：{output['regression'].item():.2%}")
    print(f"上涨概率：{output['classification'].item():.2%}")

# ===== 主程序调整 =====
if __name__ == "__main__":
    mode = 'validate'  # 可选 'train' 或 'validate'
    
    if mode == 'train':
        print("训练多任务模型...")
        train_and_save_multi_task()
    else:
        print("验证多任务模型...")
        validate_multi_task_model()
# ===== 基础库导入 =====
import torch  # PyTorch深度学习框架 示例：用于构建神经网络
import torch.nn as nn  # 神经网络模块 示例：定义全连接层nn.Linear
import torch.optim as optim  # 优化器模块 示例：使用Adam优化器
from transformers import BertTokenizer, BertModel  # BERT相关组件 示例：加载中文BERT模型
from torch.utils.data import Dataset, DataLoader  # 数据集工具 示例：创建自定义数据集
import pandas as pd  # 数据处理库 示例：处理时间序列数据
import numpy as np  # 数值计算库 示例：生成模拟数据
import re  # 正则表达式 示例：文本清洗（虽然代码中未直接使用）
from sklearn.preprocessing import MinMaxScaler  # 数据标准化 示例：将价格数据缩放到0-1
from sklearn.metrics import accuracy_score, classification_report  # 评估指标 示例：计算准确率
import joblib  # 模型保存 示例：保存特征工程的scaler
import os

# ===== 模型缓存路径配置 =====
CACHE_DIR = r"D:\Models\huggingface\hub\bert-base-chinese"  # 本地BERT模型缓存路径（示例：避免重复下载）

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

class FusionPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = NewsEncoder()  # 文本编码器实例
        self.ts_processor = TimeSeriesProcessor(input_dim=5)  # 时序处理器实例
        # 分类器（示例：拼接文本和时序特征后分类）
        self.classifier = nn.Sequential(
            nn.Linear(64 + 32, 32),  # 文本64维 + 时序32维 = 96维输入
            nn.ReLU(),               # 激活函数
            nn.Dropout(0.2),         # 随机失活
            nn.Linear(32, 1),        # 输出层
            nn.Sigmoid()             # 转换为概率（0-1）
        )

    def forward(self, text_inputs, ts_inputs):
        text_feats = self.text_encoder(**text_inputs)  # 获取文本特征
        ts_feats = self.ts_processor(ts_inputs)       # 获取时序特征
        combined = torch.cat([text_feats, ts_feats], dim=1)  # 特征拼接
        return self.classifier(combined)  # 输出预测概率

# ===== 2. 特征工程实现 =====
class TechnicalIndicators:
    @staticmethod
    def moving_average(data, window):
        """计算移动平均线（示例：5日均线）"""
        return pd.Series(data).rolling(window=window).mean()

    @staticmethod
    def rsi(data, window=14):
        """计算相对强弱指数（示例：14日RSI）"""
        series = pd.Series(data)
        delta = series.diff(1)
        gain = delta.where(delta > 0, 0)  # 计算上涨幅度
        loss = -delta.where(delta < 0, 0) # 计算下跌幅度
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

class FeatureBuilder:
    def __init__(self):
        # 初始化BERT分词器（示例：将文本转为token ID）
        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-chinese', 
            cache_dir=CACHE_DIR
        )
        self.scaler = MinMaxScaler()  # 归一化器（示例：将价格缩放到0-1）
        # 预定义的关键词词典（示例：检测新闻中的金融关键词）
        self.gold_keywords = {
            'monetary': ['降息', '加息', '利率', '货币政策'],
            'inflation': ['通胀', 'CPI', 'PPI', '物价'],
            'geopolitical': ['冲突', '战争', '制裁', '局势紧张'],
            'safe_haven': ['避险', '黄金', '美元', '国债']
        }
    
    def process_news(self, news_text):
        """处理新闻文本（示例：输入"美联储降息"返回token和关键词计数）"""
        # BERT分词（返回字典包含input_ids等）
        inputs = self.tokenizer(
            news_text,
            max_length=512,     # 最大长度
            truncation=True,    # 超长截断
            padding='max_length', # 填充到最大长度
            return_tensors='pt' # 返回PyTorch tensor
        )
        # 统计关键词出现次数（示例：统计"降息"出现次数）
        keyword_counts = {k: sum(1 for kw in v if kw in news_text) 
                          for k, v in self.gold_keywords.items()}
        return inputs, keyword_counts
    
    def process_time_series(self, price_data, window=5):
        """处理时序数据（示例：输入20天价格，输出5个技术指标）"""
        data_length = len(price_data)
        # 动态调整计算窗口（示例：数据不足时减小窗口）
        ma_short = min(window, data_length - 1)
        ma_long = min(20, data_length - 1)
        rsi_window = min(14, data_length - 1)
        
        # 创建包含技术指标的DataFrame
        ts_data = pd.DataFrame({
            'close': price_data,  # 收盘价
            'ma5': self.moving_average(price_data, ma_short),  # 5日均线
            'ma20': self.moving_average(price_data, ma_long),  # 20日均线
            'rsi': self.rsi(price_data, window=rsi_window),    # RSI指标
            'volume': np.random.randn(len(price_data))         # 模拟成交量
        })
        
        # 处理缺失值（示例：用前值填充）
        ts_data.fillna(method='ffill', inplace=True)  # 前向填充
        ts_data.fillna(method='bfill', inplace=True)  # 后向填充（处理开头NaN）
        
        return self.scaler.fit_transform(ts_data)  # 归一化后返回
    
    def build_sample(self, news_text, price_history):
        """构建训练样本（示例：组合文本和时序特征）"""
        text_inputs, keyword_feats = self.process_news(news_text)
        # 移除不需要的token_type_ids（适用于单句分类）
        text_inputs = {k: v for k, v in text_inputs.items() 
                      if k in ['input_ids', 'attention_mask']}
        ts_feats = self.process_time_series(price_history)  # 处理时序数据
        return {
            'text': text_inputs,     # BERT输入
            'ts': torch.tensor(ts_feats, dtype=torch.float32), # 时序特征
            'keywords': keyword_feats # 关键词统计（虽然未在模型中使用）
        }


# ===== 3. 数据集定义 =====
class GoldETFDataSet(Dataset):
    def __init__(self, data_list):
        """自定义数据集类（示例：包装处理后的样本数据）"""
        self.data = data_list  # 已处理的数据列表
    
    def __len__(self):
        """返回数据集大小（示例：200个样本返回200）"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """获取单个样本（示例：idx=0取第一个样本）"""
        sample = self.data[idx]
        # 调整文本输入的格式（移除batch维度）
        text_inputs = {k: v.squeeze(0) for k, v in sample['text'].items()}
        # 构造返回元组（文本输入，时序数据，标签）
        return (
            text_inputs,  # 格式：{'input_ids': tensor, 'attention_mask': tensor}
            sample['ts'],  # 时序特征 tensor
            torch.tensor(sample['label'], dtype=torch.float32).unsqueeze(0) # 标签转为tensor
        )

# ===== 4. 数据生成与加载 =====
def generate_synthetic_data(days=200, funds=2):
    """生成模拟数据（示例：创建虚拟的黄金ETF数据）"""
    np.random.seed(42)  # 固定随机种子（确保可重复性）
    data = []
    fund_names = [f"黄金ETF_{i+1}" for i in range(funds)]  # 生成基金名称
    
    for fund in fund_names:
        # 生成价格序列（示例：使用随机游走模型）
        price_history = np.cumsum(np.random.randn(days)) + 100  
        # 从第20天开始（保证有足够历史数据）
        for i in range(20, days):
            # 生成关联新闻（示例：根据价格变化生成不同新闻）
            news_text = generate_news(price_history[i-1], price_history[i])
            # 生成标签（1表示当天价格上涨）
            label = 1 if price_history[i] > price_history[i-1] else 0
            data.append({
                'fund': fund,
                'date': f'2023-01-{i:02d}',  # 生成日期字符串
                'news': news_text,  # 生成的新闻文本
                'price_history': price_history[max(0, i-20):i],  # 取最近20天价格
                'label': label  # 0/1标签
            })
    return data

def fetch_data():
    # 获取黄金ETF数据 (示例使用GLD)
    gld = yf.download('GLD', start='2010-01-01', end='2023-12-31')
    
    # 生成模拟宏观经济数据（实际使用时需连接数据库）
    dates = pd.date_range(start='2010-01-01', end='2023-12-31', freq='D')
    macro_data = pd.DataFrame(index=dates)
    macro_data['CPI'] = np.sin(np.linspace(0, 10, len(dates))) * 0.5 + 2.5  # 模拟CPI
    macro_data['DXY'] = np.cos(np.linspace(0, 8, len(dates))) * 2 + 95       # 模拟美元指数
    
    # 合并数据
    merged_data = pd.merge(gld[['Close', 'Volume']], macro_data, left_index=True, right_index=True, how='left')
    merged_data.ffill(inplace=True)  # 前向填充缺失值
    return merged_data



# ===== 5. 训练与评估 =====
def train_model(model, dataloader, epochs=10, lr=3e-5):
    """模型训练函数（示例：输入数据和模型，输出训练好的模型）"""
    criterion = nn.BCELoss()  # 二分类交叉熵损失
    optimizer = optim.AdamW(model.parameters(), lr=lr)  # 使用AdamW优化器
    model.train()  # 设置模型为训练模式
    
    for epoch in range(epochs):
        total_loss = 0.0
        for text_inputs, ts_inputs, labels in dataloader:
            optimizer.zero_grad()  # 清空梯度
            outputs = model(text_inputs, ts_inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            total_loss += loss.item()  # 累加损失值
        # 打印每个epoch的损失
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
    return model

def evaluate_model(model, dataloader):
    """模型评估函数（示例：计算准确率和分类报告）"""
    model.eval()  # 设置模型为评估模式
    y_true, y_pred = [], []  # 存储真实标签和预测结果
    
    with torch.no_grad():  # 禁用梯度计算
        for text_inputs, ts_inputs, labels in dataloader:
            outputs = model(text_inputs, ts_inputs)  # 前向传播
            # 收集结果（将tensor转为numpy）
            y_true.extend(labels.numpy().flatten())
            y_pred.extend((outputs > 0.5).numpy().flatten())  # 概率转0/1预测
    
    # 输出评估结果
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))

def validate_saved_model(model_path="gold_etf_model.pth", scaler_path="scaler.pkl"):
    """验证已保存模型的函数"""
    # 1. 加载特征处理器
    feature_builder = FeatureBuilder()
    feature_builder.scaler = joblib.load(scaler_path)

    # 2. 初始化模型并加载保存的参数
    model = FusionPredictor()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 设置为评估模式

    # 3. 准备测试数据（使用与训练相同的生成方法）
    test_samples = generate_synthetic_data(days=50)  # 生成50天的测试数据
    processed_data = []
    
    for sample in test_samples:
        features = feature_builder.build_sample(
            sample['news'],
            sample['price_history']
        )
        processed_data.append({
            'text': features['text'],
            'ts': features['ts'],
            'label': sample['label']
        })

    # 4. 创建测试数据加载器
    test_dataset = GoldETFDataSet(processed_data)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # 5. 评估模型
    print("\n加载模型评估结果：")
    evaluate_model(model, test_loader)

    # 6. 单个预测示例
    test_news = "美联储宣布降息25个基点，市场避险情绪升温"
    test_price = np.array([100, 102, 101, 103, 104])
    features = feature_builder.build_sample(test_news, test_price)

    input_text = {k: v.unsqueeze(0) for k, v in features['text'].items()}
    input_ts = features['ts'].unsqueeze(0)

    with torch.no_grad():
        output = model(input_text, input_ts)
    print(f"\n单条预测结果：上涨概率 {output.item():.2%}")

def train_and_save_model():
    """原有的训练和保存模型的完整流程"""
    # 1. 生成模拟数据（示例：创建200天的虚拟数据）
    synthetic_data = generate_synthetic_data(days=200)
    feature_builder = FeatureBuilder()
    
    # 2. 构建数据集（示例：将原始数据转为模型可用的格式）
    processed_data = []
    for sample in synthetic_data:
        features = feature_builder.build_sample(
            sample['news'], 
            sample['price_history']  # 传入价格历史数据
        )
        processed_data.append({
            'text': features['text'],    # BERT输入特征
            'ts': features['ts'],        # 时序特征
            'label': sample['label']     # 真实标签
        })
    
    # 3. 划分训练集/测试集（示例：80%训练，20%测试）
    split_idx = int(0.8 * len(processed_data))  # 计算分割点
    train_data = processed_data[:split_idx]     # 前80%训练
    test_data = processed_data[split_idx:]      # 后20%测试
    
    # 创建PyTorch数据集和数据加载器
    train_dataset = GoldETFDataSet(train_data)
    test_dataset = GoldETFDataSet(test_data)
    # 数据加载器参数说明：batch_size=8表示每次取8个样本，shuffle=False因为时序数据不能打乱
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # 4. 初始化并训练模型
    model = FusionPredictor()  # 实例化模型
    trained_model = train_model(model, train_loader)  # 训练10个epoch
    
    # 5. 模型评估
    print("\n模型评估结果：")
    evaluate_model(trained_model, test_loader)  # 在测试集上评估
    
    # 6. 保存模型和特征处理器
    model_directory = r"E:\Models\fine-tune"
    model_path = os.path.join(model_directory, "gold_etf_model.pth")
    torch.save(trained_model.state_dict(), model_path)  # 保存模型参数
    joblib.dump(feature_builder.scaler, "scaler.pkl")  # 保存归一化器
    
    # 7. 预测示例
    test_news = "美联储宣布降息25个基点，市场避险情绪升温"  # 示例新闻
    test_price = np.array([100, 102, 101, 103, 104])  # 模拟5天价格数据
    features = feature_builder.build_sample(test_news, test_price)  # 特征处理
    
    # 调整输入格式（添加batch维度）
    input_text = {k: v.unsqueeze(0) for k, v in features['text'].items()}  # batch_size=1
    input_ts = features['ts'].unsqueeze(0)  # 从(5,5)变为(1,5,5)
    
    # 执行预测
    with torch.no_grad():
        output = trained_model(input_text, input_ts)  # 模型推理
    print(f"\n预测结果：上涨概率 {output.item():.2%}")  # 格式化输出概率

# ===== 6. 主程序流程 =====
if __name__ == "__main__":
    # 选择模式：'train' 用于训练新模型，'validate' 用于验证已有模型
    mode = 'validate'  # 或 'train'
    
    if mode == 'train':
        print("开始训练新模型...")
        train_and_save_model()
    else:
        print("开始验证已保存的模型...")
        validate_saved_model()
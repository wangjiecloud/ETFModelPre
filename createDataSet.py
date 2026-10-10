import yfinance as yf
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from datetime import datetime, timedelta
import random

class GoldETFDataGenerator:
    def __init__(self):
        self.templates = [
            "美联储宣布{action}基准利率{value}个基点，市场预期{expectation}",
            "中东{event}导致油价波动，黄金{trend}情绪升温",
            "最新{data}数据显示通胀{direction}，影响黄金投资决策"
        ]
    
    def generate_news(self, prev_price, curr_price):
        action = "降息" if random.random() < 0.6 else "维持"
        event = "冲突加剧" if curr_price > prev_price else "局势缓和"
        return random.choice(self.templates).format(
            action=action,
            value=random.randint(0, 50),
            expectation="乐观" if curr_price > prev_price else "谨慎",
            event=event,
            trend="避险" if curr_price > prev_price else "抛售",
            data="CPI" if random.random() < 0.7 else "PPI",
            direction="上升" if random.random() < 0.6 else "下降"
        )

    def download_real_data(self, symbol='GLD', start='2008-01-01', end='2023-12-31'):
        df = yf.download(symbol, start=start, end=end, auto_adjust=True)
        return df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

    def calculate_technical_indicators(self, df):
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['RSI'] = self._calculate_rsi(df['Close'])
        df['next_day_price'] = df['Close'].shift(-1)
        df['target'] = (df['next_day_price'] > df['Close']).astype(int)
        return df.dropna()

    def _calculate_rsi(self, prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def create_dataset(self, df, window_size=20):
        data = []
        for i in range(window_size, len(df)-1):
            price_history = df['Close'].iloc[i-window_size:i].values
            current_row = df.iloc[i]
            news = self.generate_news(
                df['Close'].iloc[i-1],
                current_row['Close']
            )
            
            data.append({
                'date': current_row.name.strftime('%Y-%m-%d'),
                'news': news,
                'price_history': price_history.tolist(),
                'technical_features': [
                    current_row['MA5'],
                    current_row['MA20'],
                    current_row['RSI'],
                    current_row['Volume'],
                    current_row['Close']
                ],
                'label': current_row['target']
            })
        return data

def main():
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    
    # 初始化数据生成器
    generator = GoldETFDataGenerator()
    
    # 下载并处理实际数据
    df = generator.download_real_data()
    df = generator.calculate_technical_indicators(df)
    
    # 创建完整数据集
    full_dataset = generator.create_dataset(df)
    
    # 划分数据集
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    
    # 转换为pandas DataFrame
    df_full = pd.DataFrame(full_dataset)
    
    # 创建数据集字典
    dataset_dict = DatasetDict({
        "train": Dataset.from_pandas(df_full[:train_size]),
        "validation": Dataset.from_pandas(df_full[train_size:train_size+val_size]),
        "test": Dataset.from_pandas(df_full[train_size+val_size:])
    })
    
    # 保存数据集
    dataset_dict.save_to_disk("gold_etf_dataset")
    print(f"Dataset created with sizes: ")
    print(f"Train: {len(dataset_dict['train'])} samples")
    print(f"Validation: {len(dataset_dict['validation'])} samples")
    print(f"Test: {len(dataset_dict['test'])} samples")

if __name__ == "__main__":
    main()
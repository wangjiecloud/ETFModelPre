from transformers import BertTokenizer, BertModel  # BERT相关组件 示例：加载中文BERT模型
from sklearn.preprocessing import MinMaxScaler  # 数据标准化 示例：将价格数据缩放到0-1


 # ===== 模型缓存路径配置 =====
CACHE_DIR = r"D:\Models\huggingface\hub\bert-base-chinese"  # 本地BERT模型缓存路径（示例：避免重复下载）
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
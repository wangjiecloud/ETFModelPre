import numpy as np  # 数值计算库 示例：生成模拟数据
from ..datautils.utils.db_operations import DatabaseManager


class fetchData:
    def __init__(self):
       self.basicMarketData = self.basicMarketData()    
       self.associatedAssetData = self.associatedAssetData()
    #    self.basicMarketCountData = len(self.basicMarketData)

    def generate_synthetic_data(self):
        """生成包含收益率标签的数据"""
        np.random.seed(42)
        data = []
        
        for price_history in self.basicMarketData:
             for gold_price_history in self.associatedAssetData:
                # news_text = generate_news(price_history[i-1], price_history[i])
                if price_history['trade_date'] == gold_price_history['trade_date']:
                    # news_text = generate_news(price_history['close'], gold_price_history['close'])
                    # 计算实际收益率（回归目标）
                    return_rate = price_history['change_percent']
                    
                    # 生成分类标签（1表示上涨）
                    trend_label = 1 if return_rate > 0 else 0
                    
                    data.append({
                        'date': price_history['trade_date'],
                        'news': news_text,
                        'basic_etf_Data': price_history,
                        'basic_gold_data': gold_price_history,
                        'return_rate': return_rate,  # 新增回归目标
                        'trend_label': trend_label   # 原分类目标
                    })
        return data
    
    #基础市场数据
    def basicMarketData():
        """获取基础市场数据（示例：黄金ETF的历史价格）"""
        result = DatabaseManager.query_etf_daily_data(
            start_date='2001-01-01', 
            end_date='2028-12-31'
        )
        return result
     
    # 关联资产数据
    def associatedAssetData():
        """现货黄金价格：伦敦金（LBMA Gold Price）"""
        result = DatabaseManager.query_goldprice_daily_data(
            start_date='2001-01-01', 
            end_date='2028-12-31'
        )
        return result
    
    # VIX恐慌指数回落、黄金ETF持仓量骤减、美元指数反弹

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

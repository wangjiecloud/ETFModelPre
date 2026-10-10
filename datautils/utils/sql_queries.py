# sql_queries.py

# 创建日线数据表
CREATE_DAILY_TABLE = """
 CREATE TABLE IF NOT EXISTS etf_daily_data (
    id INT AUTO_INCREMENT PRIMARY KEY,  -- 主键ID
    trade_date DATE NOT NULL,       -- 交易日期
    open_price DECIMAL(10, 3),      -- 开盘价
    high_price DECIMAL(10, 3),      -- 最高价
    low_price DECIMAL(10, 3),       -- 最低价
    close_price DECIMAL(10, 3),       -- 收盘价
    volume BIGINT,               -- 成交量
    amount DECIMAL(16, 2),       -- 成交额
    amplitude DECIMAL(10, 3),    -- 振幅
    price_change DECIMAL(10, 3), -- 涨跌
    change_percent DECIMAL(10, 3), -- 涨幅
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP  -- 创建时间
)
"""

# 创建分钟数据表
CREATE_MINUTE_TABLE = """
CREATE TABLE IF NOT EXISTS etf_minute_data (
    id INT AUTO_INCREMENT PRIMARY KEY,  -- 主键ID
    trade_date DATE NOT NULL,       -- 交易日期
    trade_time CHAR(4) NOT NULL,
    open_price DECIMAL(10, 3),      -- 开盘价
    high_price DECIMAL(10, 3),      -- 最高价
    low_price DECIMAL(10, 3),       -- 最低价
    close_price DECIMAL(10, 3),       -- 收盘价
    volume BIGINT,               -- 成交量
    amount DECIMAL(16, 2),       -- 成交额
    amplitude DECIMAL(10, 3),    -- 振幅
    price_change DECIMAL(10, 3), -- 涨跌
    change_percent DECIMAL(10, 3), -- 涨幅
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

# 创建期货黄金历史数据表
CREATE_GOLD_DAILY_TABLE = """
CREATE TABLE IF NOT EXISTS gold_daily_data (
    id INT AUTO_INCREMENT PRIMARY KEY,  -- 主键ID
    trade_date DATE NOT NULL,       -- 交易日期
    open_price DECIMAL(10, 3),      -- 开盘价
    high_price DECIMAL(10, 3),      -- 最高价
    low_price DECIMAL(10, 3),       -- 最低价
    close_price DECIMAL(10, 3),       -- 收盘价
    volume BIGINT,               -- 成交量
    amplitude DECIMAL(10, 3),    -- 振幅
    price_change DECIMAL(10, 3), -- 涨跌
    change_percent DECIMAL(10, 3), -- 涨幅
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

# 插入黄金期货历史日线数据
INSERT_GOLD_DAILY_DATA = """
INSERT INTO gold_daily_data 
(trade_date, open_price, high_price, low_price, close_price, 
    volume, amplitude, price_change, change_percent)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

# 插入ETF日线数据
INSERT_DAILY_DATA = """
INSERT INTO etf_daily_data 
(trade_date, open_price, high_price, low_price, close_price, 
    volume, amount, amplitude, price_change, change_percent)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

# 插入分钟数据
INSERT_MINUTE_DATA = """
INSERT INTO etf_minute_data 
(trade_date, trade_time, open_price, high_price, low_price, close_price, volume, amount, amplitude, price_change, change_percent)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""
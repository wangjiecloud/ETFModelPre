import mysql.connector
from mysql.connector import Error
from typing import List, Dict, Any
from .sql_queries import *


class DatabaseManager:
    def __init__(self, host: str, user: str, password: str, database: str):
        """初始化数据库连接"""
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None
        self.cursor = None

    def connect(self) -> bool:
        """建立数据库连接"""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            if self.connection.is_connected():
                self.cursor = self.connection.cursor()
                return True
        except Error as e:
            print(f"Error connecting to MySQL: {e}")
            return False
        
    def disconnect(self):
        """关闭数据库连接"""
        if self.connection and self.connection.is_connected():
            if self.cursor:
                self.cursor.close()
            self.connection.close()
    
    def create_tables(self):
        try:
            self.cursor.execute(CREATE_GOLD_DAILY_TABLE)
            self.cursor.execute(CREATE_DAILY_TABLE)
            self.cursor.execute(CREATE_MINUTE_TABLE)
            self.connection.commit()
            return True
        except Error as e:
            print(f"Error creating table: {e}")
            return False

    def insert_etf_daily_data(self, data: List[Dict[str, Any]]) -> bool:
        """插入ETF数据"""
        try:
            # 按日期排序数据以获取前一天的收盘价
            sorted_data = sorted(data, key=lambda x: x['trade_date'])
            # 初始化前一天收盘价
            prev_close = None
            processed_data = []
            for row in sorted_data:
                if prev_close is not None:
                    # 计算振幅、涨跌和涨幅
                    amplitude = ((row['high_price'] - row['low_price']) / prev_close) * 100
                    price_change = row['close_price'] - prev_close
                    change_percent = (price_change / prev_close) * 100
                else:
                    # 第一条数据，设置为0
                    amplitude = 0
                    price_change = 0
                    change_percent = 0
                processed_data.append((
                    row['trade_date'],
                    row['open_price'],
                    row['high_price'],
                    row['low_price'],
                    row['close_price'],
                    row['volume'],
                    row['amount'],
                    amplitude,
                    price_change,
                    change_percent
                ))
                prev_close = row['close_price']
            insert_query = INSERT_DAILY_DATA
            self.cursor.executemany(insert_query, processed_data)
            self.connection.commit()
            return True
        except Error as e:
            print(f"Error inserting data: {e}")
            return False
    
    def insert_gold_daily_data(self, data: List[Dict[str, Any]]) -> bool:
        """插入ETF数据"""
        try:
            # 按日期排序数据以获取前一天的收盘价
            sorted_data = sorted(data, key=lambda x: x['trade_date'])
            # 初始化前一天收盘价
            prev_close = None
            processed_data = []
            for row in sorted_data:
                if prev_close is not None:
                    # 计算振幅、涨跌和涨幅
                    amplitude = ((row['high_price'] - row['low_price']) / prev_close) * 100
                    price_change = row['close_price'] - prev_close
                else:
                    # 第一条数据，设置为0
                    amplitude = 0
                    price_change = 0
                processed_data.append((
                    row['trade_date'],
                    row['open_price'],
                    row['high_price'],
                    row['low_price'],
                    row['close_price'],
                    row['volume'],
                    amplitude,
                    price_change,
                    row['change_percent']
                ))
                prev_close = row['close_price']
            insert_query = INSERT_GOLD_DAILY_DATA
            self.cursor.executemany(insert_query, processed_data)
            self.connection.commit()
            return True
        except Error as e:
            print(f"Error inserting data: {e}")
            return False
    def insert_etf_min_data(self, data: List[Dict[str, Any]]) -> bool:
        try:
            # 按日期排序数据以获取前一天的收盘价
            sorted_data = sorted(data, key=lambda x: x['trade_date'])
            sorted_data = sorted(data, key=lambda x: (x['trade_date'], x['trade_time']))
            # 初始化前一天收盘价
            prev_close = None
            processed_data = []
            for row in sorted_data:
                if prev_close is not None:
                    # 计算振幅、涨跌和涨幅
                    amplitude = ((row['high_price'] - row['low_price']) / prev_close) * 100
                    price_change = row['close_price'] - prev_close
                    change_percent = (price_change / prev_close) * 100
                else:
                    # 第一条数据，设置为0
                    amplitude = 0
                    price_change = 0
                    change_percent = 0
                processed_data.append((
                    row['trade_date'],
                    row['trade_time'],
                    row['open_price'],
                    row['high_price'],
                    row['low_price'],
                    row['close_price'],
                    row['volume'],
                    row['amount'],
                    amplitude,
                    price_change,
                    change_percent
                ))
                prev_close = row['close_price']
            insert_query = INSERT_MINUTE_DATA
            self.cursor.executemany(insert_query, processed_data)
            self.connection.commit()
            return True
        except Error as e:
            print(f"Error inserting data: {e}")
            return False

    def query_etf_daily_data(self, start_date: str = None, end_date: str = None) -> List[Dict[str, Any]]:
        """查询ETF数据"""
        try:
            query = "SELECT * FROM etf_daily_data WHERE 1=1"
            params = []
            
            if start_date:
                query += " AND trade_date >= %s"
                params.append(start_date)
            if end_date:
                query += " AND trade_date <= %s"
                params.append(end_date)
                
            query += " ORDER BY trade_date"
            
            self.cursor.execute(query, params)
            columns = [desc[0] for desc in self.cursor.description]
            result = []
            
            for row in self.cursor.fetchall():
                result.append(dict(zip(columns, row)))
            
            return result
        except Error as e:
            print(f"Error querying data: {e}")
            return []
    
    def query_goldprice_daily_data(self, start_date: str = None, end_date: str = None) -> List[Dict[str, Any]]:
        try:
            query = "SELECT * FROM gold_daily_data WHERE 1=1"
            params = []
            
            if start_date:
                query += " AND trade_date >= %s"
                params.append(start_date)
            if end_date:
                query += " AND trade_date <= %s"
                params.append(end_date)
                
            query += " ORDER BY trade_date"
            
            self.cursor.execute(query, params)
            columns = [desc[0] for desc in self.cursor.description]
            result = []
            
            for row in self.cursor.fetchall():
                result.append(dict(zip(columns, row)))
            
            return result
        except Error as e:
            print(f"Error querying data: {e}")
            return []

    def update_etf_data(self, trade_date: str, data: Dict[str, Any]) -> bool:
        """更新ETF数据"""
        try:
            # 获取前一天的收盘价
            prev_day_query = """
            SELECT close_price FROM etf_data 
            WHERE trade_date < %s 
            ORDER BY trade_date DESC LIMIT 1
            """
            self.cursor.execute(prev_day_query, (trade_date,))
            prev_close = self.cursor.fetchone()
            
            if prev_close:
                prev_close = prev_close[0]
                # 计算新字段
                amplitude = ((data['high_price'] - data['low_price']) / prev_close) * 100
                price_change = data['close_price'] - prev_close
                change_percent = (price_change / prev_close) * 100
            else:
                amplitude = 0
                price_change = 0
                change_percent = 0

            update_query = """
            UPDATE etf_data 
            SET open_price = %s, high_price = %s, low_price = %s, 
                close_price = %s, volume = %s, amount = %s,
                amplitude = %s, price_change = %s, change_percent = %s
            WHERE trade_date = %s
            """
            self.cursor.execute(update_query, (
                data['open_price'],
                data['high_price'],
                data['low_price'],
                data['close_price'],
                data['volume'],
                data['amount'],
                amplitude,
                price_change,
                change_percent,
                trade_date
            ))
            self.connection.commit()
            return True
        except Error as e:
            print(f"Error updating data: {e}")
            return False

    def delete_etf_data(self, trade_date: str) -> bool:
        """删除ETF数据"""
        try:
            delete_query = "DELETE FROM etf_data WHERE trade_date = %s"
            self.cursor.execute(delete_query, (trade_date,))
            self.connection.commit()
            return True
        except Error as e:
            print(f"Error deleting data: {e}")
            return False
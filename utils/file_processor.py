import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from datetime import datetime
from typing import List, Dict, Any
from datautils.utils.db_operations import DatabaseManager


class ETFFileProcessor:
    def __init__(self, db_manager: DatabaseManager):
        """初始化文件处理器"""
        self.db_manager = db_manager

    def parse_line(self, line: str, flag: str) -> Dict[str, Any]:
        """解析单行数据"""
        try:
            # 分割行数据
            parts = line.strip().split('\t').split(',')

            # 处理日期
            date_str = parts[0].replace('/', '-').replace('"', "")
            trade_date = datetime.strptime(date_str, '%Y-%m-%d').date()

            # 处理数值数据
            def parse_number(value: str) -> float:
                try:
                    return float(value) if value.strip() else None
                except ValueError:
                    return None

            if flag == 'daily':
                # 返回解析后的数据字典
                return {
                    'trade_date': trade_date,
                    'open_price': parse_number(parts[1]),
                    'high_price': parse_number(parts[2]),
                    'low_price': parse_number(parts[3]),
                    'close_price': parse_number(parts[4]),
                    'volume': int(float(parts[5])) if len(parts) > 5 and parts[5].strip() else None,
                    'amount': parse_number(parts[6]) if len(parts) > 6 else None
                }
            elif flag == 'min':
                return {
                    'trade_date': trade_date,
                    'trade_time': parts[1],
                    'open_price': parse_number(parts[2]),
                    'high_price': parse_number(parts[3]),
                    'low_price': parse_number(parts[4]),
                    'close_price': parse_number(parts[5]),
                    'volume': int(float(parts[6])) if len(parts) > 5 and parts[6].strip() else None,
                    'amount': parse_number(parts[7]) if len(parts) > 6 else None
                }
            elif flag == 'gold_daily':
                return {
                    'trade_date': trade_date,
                    'open_price': parse_number(parts[2].replace('"', "")),
                    'high_price': parse_number(parts[3].replace('"', "")),
                    'low_price': parse_number(parts[4].replace('"', "")),
                    'close_price': parse_number(parts[1].replace('"', "")),
                    'volume': self.convert_k_to_num(parts[5].replace('"', "")),
                    'change_percent': self.convert_percent_string_keep(parse_number(parts[6].replace('"', "")))
                }
        except Exception as e:
            print(f"Error parsing line: {line}")
            print(f"Error details: {e}")
            return None
    def convert_k_to_num(s):
        s = s.upper()
        if 'K' in s:
            num_str = s.replace('K', '')
            return float(num_str) * 1000
        else:
            return float(s)
    def convert_percent_string_keep(s):
        import re
        match = re.match(r'^\s*([-+]?\d+(?:\.\d+)?)\s*%\s*$', s, re.IGNORECASE)
        if match:
            return float(match.group(1))  # 直接返回百分比数值（如 -2.76）
        else:
            raise ValueError("无效的百分比格式")
    def insert_data(self, file_path, data_type):
        data_to_insert = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.startswith('#') or not line.strip():
                    continue
                parsed_data = self.parse_line(line, data_type)
                if parsed_data:
                    data_to_insert.append(parsed_data)
        
        if data_to_insert:
            if data_type == 'min':
                success = self.db_manager.insert_etf_min_data(data_to_insert)
            elif data_type == 'daily':
                success = self.db_manager.insert_etf_daily_data(data_to_insert)
            elif data_type == 'gold_daily':
                success = self.db_manager.insert_gold_daily_data(data_to_insert)
            else:
                print(f"Unsupported data type: {data_type}")
                return False
            
            if success:
                print(f"Successfully inserted {len(data_to_insert)} records")
            return success
        return False

    def process_file(self, file_path1: str, file_path2: str, file_path3: str) -> bool:
        """处理ETF数据文件"""
        try:
            if not self.db_manager.connect():
                return False

            if not self.db_manager.create_tables():
                return False

            # Corrected method call
            self.insert_data(file_path1, 'daily')
            self.insert_data(file_path2, 'min')
            self.insert_data(file_path3, 'gold_daily')

        except Exception as e:
            print(f"Error processing file: {e}")
            return False

        finally:
            self.db_manager.disconnect()

def main():
    """主函数"""
    # 数据库配置
    db_config = {
        'host': 'localhost',
        'user': 'wj',
        'password': '123456',
        'database': 'ETFS'
    }

    # 创建数据库管理器和文件处理器
    db_manager = DatabaseManager(**db_config)
    file_processor = ETFFileProcessor(db_manager)

    # 处理文件
    file_path1 = r'C:\workspace\SuperJAI\projects\ETFModelPre\datautils\datasets\SH#518850.txt'
    file_path2 = r'C:\workspace\SuperJAI\projects\ETFModelPre\datautils\datasets\SH#f518850.txt'
    file_path3 = r'C:\workspace\SuperJAI\projects\ETFModelPre\datautils\datasets\黄金期货历史数据.csf'

    success = file_processor.process_file(file_path1,file_path2,file_path3)

    if success:
        print("File processing completed successfully")
    else:
        print("File processing failed")

if __name__ == "__main__":
    main()
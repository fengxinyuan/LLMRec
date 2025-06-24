import json
import re
from llm_api import llm_extend, llm_categorize_and_recommend

def load_json_data(filepath):
    """加载JSON文件"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误: 文件未找到 {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"错误: 文件 {filepath} JSON解码失败")
        return None

def initialize_json_file(filepath):
    """初始化一个空的JSON列表文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump([], f)

def append_to_json_file(data, filepath):
    """将单个JSON对象追加到文件中"""
    with open(filepath, 'r+', encoding='utf-8') as f:
        # 移动到文件末尾的前一个位置，以覆盖 ']'
        f.seek(0, 2)
        f.seek(f.tell() - 1, 0)
        
        # 如果文件不为空，则在追加前添加逗号
        if f.tell() > 1:
            f.write(',')
            
        # 追加新数据并关闭列表
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write(']')

from llm_api import llm_format_and_enhance_user_profile

def process_users_with_llm(users, output_filepath):
    """使用LLM格式化和增强用户数据，并逐个保存"""
    for i, user in enumerate(users):
        print(f"正在处理用户 {i+1}/{len(users)}: {user.get('user_id', 'N/A')}")
        
        user_json_string = json.dumps(user, ensure_ascii=False)
        
        # 调用LLM进行格式化和增强
        processed_profile = llm_format_and_enhance_user_profile(user_json_string)
        
        if processed_profile:
            append_to_json_file(processed_profile, output_filepath)
            print(f"用户 {user.get('user_id', 'N/A')} 处理并保存成功。")
        else:
            print(f"警告: 未能处理用户 {user.get('user_id', 'N/A')} 的数据，将跳过此用户。")

def main():
    """主执行函数"""
    output_filepath = 'users_processed.json'
    
    # 初始化输出文件
    initialize_json_file(output_filepath)
    
    # 格式化和增强用户数据
    users_data = load_json_data('users.json')
    if users_data:
        process_users_with_llm(users_data, output_filepath)
        print(f"\n所有用户处理完成。结果已保存到 {output_filepath}")

    # 接下来可以添加处理知识数据的逻辑
    # knowledge_data = load_json_data('knowledge.json')
    # if knowledge_data:
    #     # ... 知识数据处理 ...
    #     pass

if __name__ == "__main__":
    main()
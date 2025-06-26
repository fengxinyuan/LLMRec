import json
import os
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

def generate_html_report(user_profile, articles_dir='articles'):
    """为单个用户生成HTML报告"""
    user_id = user_profile.get('user_id', 'N/A')
    print(f"正在为用户 {user_id} 生成报告...")

    # 确保articles目录存在
    if not os.path.exists(articles_dir):
        os.makedirs(articles_dir)

    # 调用LLM进行内容扩展和推荐
    extended_content = llm_extend(json.dumps(user_profile, ensure_ascii=False))
    
    if not extended_content:
        print(f"警告: 未能为用户 {user_id} 生成扩展内容。")
        return

    # 假设llm_extend返回的是HTML格式的字符串
    html_content = extended_content

    # # 生成HTML文件名
    # report_filename = f"NEW_RES_{user_id}.html"
    # report_filepath = os.path.join(articles_dir, report_filename)

    # # 保存HTML文件
    # with open(report_filepath, 'w', encoding='utf-8') as f:
    #     f.write(html_content)
    
    # print(f"成功为用户 {user_id} 生成报告: {report_filepath}")

def main():
    """主函数：加载处理过的用户数据并生成报告"""
    processed_users_filepath = 'users_processed.json'
    
    users_data = load_json_data(processed_users_filepath)
    
    if users_data:
        for user in users_data:
            generate_html_report(user)
        print("\n所有用户报告生成完成。")

if __name__ == "__main__":
    main()
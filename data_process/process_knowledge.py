import json
import os
from llm_api import (
    llm_format_and_enhance_knowledge_resource,
    llm_generate_html_for_knowledge,
    llm_generate_new_knowledge_resources
)

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

def save_json_data(data, filepath):
    """将数据保存为JSON文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"数据已成功保存到 {filepath}")

def save_html_content(html_string, filepath):
    """将HTML字符串保存到文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html_string)
    print(f"HTML文件已保存到 {filepath}")

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

def optimize_knowledge_resources(knowledge_data):
    """使用LLM优化现有的知识资源"""
    optimized_resources = []
    for i, resource in enumerate(knowledge_data):
        print(f"正在优化资源 {i+1}/{len(knowledge_data)}: {resource.get('resource_id', 'N/A')}")
        resource_json = json.dumps(resource, ensure_ascii=False)
        enhanced_resource_str = llm_format_and_enhance_knowledge_resource(resource_json)
        if enhanced_resource_str:
            try:
                enhanced_resource = json.loads(enhanced_resource_str)
                optimized_resources.append(enhanced_resource)
            except json.JSONDecodeError:
                print(f"错误: 解析资源 {resource.get('resource_id', 'N/A')} 的优化结果失败。")
        else:
            print(f"未能优化资源 {resource.get('resource_id', 'N/A')}。")
    return optimized_resources

def main():
    """主函数：加载、优化并保存知识资源"""
    input_filepath = 'data/knowledge.json'
    output_filepath = 'data/knowledge_processed.json'

    knowledge_data = load_json_data(input_filepath)
    if knowledge_data:
        optimized_data = optimize_knowledge_resources(knowledge_data)
        if optimized_data:
            save_json_data(optimized_data, output_filepath)
            print(f"所有现有资源优化完成，已保存至 {output_filepath}")
        else:
            print("没有资源被成功优化。")

if __name__ == "__main__":
    main()
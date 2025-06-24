import json
import os
from llm_api import llm_generate_html_for_knowledge

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

def save_html_content(html_string, filepath):
    """将HTML字符串保存到文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html_string)
    print(f"  HTML文件已保存到: {filepath}")


def main():
    """主函数：为知识库中的资源生成HTML文章"""
    processed_knowledge_filepath = 'data/knowledge_processed.json'
    articles_dir = 'data/articles'

    # 确保articles目录存在
    if not os.path.exists(articles_dir):
        os.makedirs(articles_dir)

    # 1. 加载知识库
    knowledge_data = load_json_data(processed_knowledge_filepath)
    if not knowledge_data:
        print("错误: 无法加载知识库文件，或文件为空。程序终止。")
        return

    print(f"开始为 {len(knowledge_data)} 条知识资源生成文章...")
    generated_count = 0
    skipped_count = 0

    # 2. 遍历所有资源，检查并生成HTML文件
    for i, resource in enumerate(knowledge_data):
        resource_id = resource.get("resource_id")
        title = resource.get("title", "无标题")

        if not resource_id:
            print(f"警告: 第 {i+1} 条资源缺少 'resource_id'，已跳过。")
            continue

        html_filepath = os.path.join(articles_dir, f"{resource_id}.html")

        # 检查HTML文件是否已存在
        if os.path.exists(html_filepath):
            # print(f"文章 '{resource_id}.html' 已存在，跳过。")
            skipped_count += 1
            continue
        
        print(f"({i+1}/{len(knowledge_data)}) 正在为资源 '{title}' ({resource_id}) 生成HTML...")
        
        # a. 调用LLM生成HTML内容
        resource_json_string = json.dumps(resource, ensure_ascii=False)
        html_content = llm_generate_html_for_knowledge(resource_json_string)

        # b. 保存HTML文件
        if html_content:
            save_html_content(html_content, html_filepath)
            generated_count += 1
        else:
            print(f"  警告: 未能为资源 {resource_id} 生成HTML。")

    print(f"\n任务完成！")
    print(f"  - 新生成了 {generated_count} 篇文章。")
    print(f"  - 跳过了 {skipped_count} 篇已存在的文章。")

if __name__ == "__main__":
    main()
import json
import os
from llm_api import (
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

def append_to_json_file(data, filepath):
    """将单个JSON对象追加到文件中"""
    # 先读
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        content = []
    
    # 追加
    content.append(data)

    # 再写
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(content, f, ensure_ascii=False, indent=2)


def save_html_content(html_string, filepath):
    """将HTML字符串保存到文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html_string)
    print(f"  HTML文件已保存到: {filepath}")


def main():
    """主函数：以迭代、增量的方式扩展知识库并生成报告"""
    processed_knowledge_filepath = 'data/knowledge_processed.json'
    articles_dir = 'data/articles'
    total_new_resources_to_generate = 200
    batch_size = 5  # 每次调用LLM生成的数量

    # 确保articles目录存在
    if not os.path.exists(articles_dir):
        os.makedirs(articles_dir)

    generated_count = 0
    while generated_count < total_new_resources_to_generate:
        print(f"\n--- 开始新一轮生成 (当前: {generated_count}/{total_new_resources_to_generate}) ---")
        
        # 1. 加载最新的知识库概览
        all_processed_data = load_json_data(processed_knowledge_filepath)
        if not all_processed_data:
            print("错误: 无法加载知识库文件，程序终止。")
            return
            
        existing_summary = [{"title": r.get("title"), "technical_domain": r.get("technical_domain")} for r in all_processed_data]
        existing_summary_json = json.dumps(existing_summary, ensure_ascii=False)

        # 2. 调用LLM生成一小批新资源
        print(f"正在调用LLM生成 {batch_size} 条新资源...")
        new_resources_batch = llm_generate_new_knowledge_resources(existing_summary_json, num_to_generate=batch_size)

        if not new_resources_batch:
            print("警告: LLM未能返回新资源，稍后重试...")
            continue

        # 3. 逐个处理并保存新生成的资源
        for resource in new_resources_batch:
            if generated_count >= total_new_resources_to_generate:
                break # 避免超出目标数量

            resource_id = resource.get('resource_id', f'NEW_UNKNOWN_{generated_count}')
            print(f"处理新资源 ({generated_count + 1}/{total_new_resources_to_generate}): {resource.get('title', '无标题')}")

            # a. 追加到JSON文件
            append_to_json_file(resource, processed_knowledge_filepath)
            print(f"  已追加到 {processed_knowledge_filepath}")

            # b. 生成并保存HTML
            resource_json_string = json.dumps(resource, ensure_ascii=False)
            html_content = llm_generate_html_for_knowledge(resource_json_string)
            if html_content:
                html_filepath = os.path.join(articles_dir, f"{resource_id}.html")
                save_html_content(html_content, html_filepath)
            else:
                print(f"  警告: 未能为资源 {resource_id} 生成HTML。")
            
            generated_count += 1

    print(f"\n任务完成！总共生成了 {generated_count} 条新知识资源。")

if __name__ == "__main__":
    main()
import json
import os
from llm_api import llm_generate_new_knowledge_resources

def load_json_data(filepath):
    """加载JSON文件"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误: 文件未找到 {filepath}")
        return []
    except json.JSONDecodeError:
        print(f"错误: 文件 {filepath} JSON解码失败")
        return []

def append_to_json_file(data, filepath):
    """将单个JSON对象追加到文件中"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        content = []
    
    content.append(data)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(content, f, ensure_ascii=False, indent=2)

def main():
    """主函数：以迭代、增量的方式扩展知识库"""
    processed_knowledge_filepath = 'data/knowledge_processed.json'
    total_new_resources_to_generate = 200
    batch_size = 5  # 每次调用LLM生成的数量

    # 计算已有的资源数量，以确定还需要生成多少
    all_processed_data = load_json_data(processed_knowledge_filepath)
    generated_count = len(all_processed_data)
    
    print(f"知识库中已有 {generated_count} 条资源。目标总数: {total_new_resources_to_generate}。")

    while generated_count < total_new_resources_to_generate:
        print(f"\n--- 开始新一轮生成 (当前: {generated_count}/{total_new_resources_to_generate}) ---")
        
        # 1. 加载最新的知识库概览
        # 每次循环都重新加载，以防其他进程修改了文件
        current_data = load_json_data(processed_knowledge_filepath)
        existing_summary = [{"title": r.get("title"), "technical_domain": r.get("technical_domain")} for r in current_data]
        existing_summary_json = json.dumps(existing_summary, ensure_ascii=False)

        # 2. 计算本次需要生成的数量
        remaining_to_generate = total_new_resources_to_generate - generated_count
        current_batch_size = min(batch_size, remaining_to_generate)

        # 3. 调用LLM生成一小批新资源
        print(f"正在调用LLM生成 {current_batch_size} 条新资源...")
        new_resources_batch = llm_generate_new_knowledge_resources(existing_summary_json, num_to_generate=current_batch_size)

        if not new_resources_batch:
            print("警告: LLM未能返回新资源，稍后重试...")
            continue

        # 4. 逐个处理并保存新生成的资源
        for resource in new_resources_batch:
            if generated_count >= total_new_resources_to_generate:
                break

            # 简单校验一下资源格式
            if not isinstance(resource, dict) or 'title' not in resource:
                print(f"警告: LLM返回的资源格式不正确，已跳过: {resource}")
                continue

            # 为新资源分配一个临时ID，后续由 rename_knowledge_ids.py 统一处理
            resource['resource_id'] = f"TEMP_{generated_count + 1}"
            print(f"处理新资源 ({generated_count + 1}/{total_new_resources_to_generate}): {resource.get('title', '无标题')}")

            # a. 追加到JSON文件
            append_to_json_file(resource, processed_knowledge_filepath)
            print(f"  已追加到 {processed_knowledge_filepath}")
            
            generated_count += 1

    print(f"\n任务完成！知识库扩展至 {generated_count} 条资源。")

if __name__ == "__main__":
    main()
import json
import os

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

def rename_knowledge_ids(input_filepath, output_filepath):
    """
    重新命名知识库中的 resource_id，并更新相关的 related_resources。
    """
    knowledge_data = load_json_data(input_filepath)
    if not knowledge_data:
        return

    id_map = {}
    new_knowledge_data = []
    
    # 第一遍：创建新的唯一ID并建立新旧ID的映射
    for i, resource in enumerate(knowledge_data):
        old_id = resource.get("resource_id")
        new_id = f"RES{i+1:03}"
        
        if old_id:
            id_map[old_id] = new_id
        
        # 创建一个新的字典，避免在迭代时修改
        new_resource = resource.copy()
        new_resource["resource_id"] = new_id
        new_knowledge_data.append(new_resource)

    # 第二遍：更新 related_resources 字段
    for resource in new_knowledge_data:
        if "related_resources" in resource and isinstance(resource["related_resources"], list):
            updated_related = []
            for related_id in resource["related_resources"]:
                # 如果旧ID在映射中，则使用新ID，否则保留原样（可能指向外部资源）
                updated_related.append(id_map.get(related_id, related_id))
            resource["related_resources"] = updated_related
            
    # 保存更新后的数据
    save_json_data(new_knowledge_data, output_filepath)
    print("所有 resource_id 已成功重命名，并且 related_resources 已更新。")

def main():
    """主函数"""
    input_filepath = 'data/knowledge_processed.json'
    output_filepath = 'data/renamed_knowledge_processed.json'
    
    rename_knowledge_ids(input_filepath, output_filepath)

if __name__ == "__main__":
    main()
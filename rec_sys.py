import json
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import faiss
import numpy as np
from llm_api import llm_extend, llm_categorize_and_recommend
from sentence_transformers import SentenceTransformer

def values_to_sentence(json_data):
    values = []
    for key, value in json_data.items():
        # 根据值的类型转换为字符串
        if isinstance(value, bool):
            values.append("是" if value else "否")
        elif isinstance(value, (int, float)):
            values.append(str(value))
        elif isinstance(value, list):
            values.append(f"[{', '.join(map(str, value))}]")
        else:
            values.append(value)
    return ", ".join(values) + "。"

def list_to_sentences(json_list):
    sentences = []
    for item in json_list:
        sentences.append(values_to_sentence(item))
    return sentences

def get_recommendation(user_id):
    """
    为指定用户ID生成推荐。
    """
    print("="*20)
    print(f"开始为用户 {user_id} 生成推荐...")

    # --- 1. 加载数据 ---
    print("[1/6] 正在加载用户和知识数据...")
    with open('user-portraits/users.json', 'r', encoding='utf-8') as file:
        user_prts = json.load(file)
    with open('knowledge-protraits/knowledge.json', 'r', encoding='utf-8') as file:
        knowledge_prts = json.load(file)
    print("数据加载完毕。")

    # --- 2. 加载模型 ---
    print("[2/6] 正在加载Bi-Encoder模型...")
    bi_encoder = SentenceTransformer('bert-base-nli-mean-tokens', device='cpu')
    print("模型加载完毕。")

    # --- 3. 准备句子 ---
    print("[3/6] 正在准备用户画像和知识库句子...")
    user_info_raw = values_to_sentence(user_prts[user_id])
    print("    > 正在调用LLM增强用户画像...")
    user_info_enhanced = llm_extend(user_info_raw)
    knowledge_sentences = list_to_sentences(knowledge_prts)
    print("句子准备完毕。")

    # --- 4. 向量化与索引 ---
    print("[4/6] 正在进行知识库向量化并构建Faiss索引...")
    dimension = bi_encoder.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(dimension)
    knowledge_embeddings = bi_encoder.encode(knowledge_sentences)
    faiss.normalize_L2(knowledge_embeddings)
    index.add(knowledge_embeddings)
    print("Faiss索引构建完毕。")

    # --- 5. 召回 ---
    print("[5/6] 正在进行向量召回...")
    user_embedding = bi_encoder.encode(user_info_enhanced, convert_to_tensor=True).unsqueeze(0)
    user_embedding_np = user_embedding.cpu().numpy()
    faiss.normalize_L2(user_embedding_np)
    distances, indices = index.search(user_embedding_np, 20) # 召回10个用于rerank
    print(f"召回了 {len(indices[0])} 个候选结果。")

    # --- 6. 推荐类别生成 ---
    print("[6/6] 正在使用LLM生成推荐类别和内容...")
    retrieved_indices = indices[0]
    
    # 准备带有元数据的候选列表
    candidates_with_meta = []
    for idx in retrieved_indices:
        candidates_with_meta.append({
            "original_index": idx,
            "domain": knowledge_prts[idx].get("technical_domain", "未知领域"),
            "sentence": knowledge_sentences[idx]
        })

    # 调用LLM进行分类和推荐
    categorized_recommendations = llm_categorize_and_recommend(user_info_enhanced, candidates_with_meta)

    # 构建最终的结构化输出
    final_recommendations = {}
    for category, recommended_indices in categorized_recommendations.items():
        final_recommendations[category] = []
        for rec_idx in recommended_indices:
            # rec_idx 是 candidates_with_meta 列表的索引
            original_knowledge_index = candidates_with_meta[rec_idx]["original_index"]
            final_recommendations[category].append(knowledge_prts[original_knowledge_index])

    print("推荐类别生成完毕。")
    print("="*20)

    output = {
        "user_info": user_info_raw,
        "recommendations": final_recommendations
    }
    return output

if __name__ == "__main__":
    target_user_id = 1  # 指定要推荐的用户ID
    recommendation_result = get_recommendation(target_user_id)
    
    print("\n--- 最终推荐结果 ---")
    print(f"用户画像: {recommendation_result['user_info']}")
    print("\n--- 为您推荐的知识类别 ---")
    if recommendation_result['recommendations']:
        for category, items in recommendation_result['recommendations'].items():
            print(f"\n【{category}】")
            for item in items:
                print(f"  - {item['title']} (ID: {item['resource_id']})")
    else:
        print("未能生成有效的推荐类别。")

    # print("\n--- 完整JSON输出 ---")
    # print(json.dumps(recommendation_result, ensure_ascii=False, indent=4))
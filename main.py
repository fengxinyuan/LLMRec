import json
import os
import time
import faiss
import numpy as np
from llm_api import llm_extend, llm_categorize_and_recommend
from sentence_transformers import SentenceTransformer

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
CACHE_DIR = 'cache'
os.makedirs(CACHE_DIR, exist_ok=True)

def values_to_sentence(json_data):
    values = []
    for key, value in json_data.items():
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
    return [values_to_sentence(item) for item in json_list]

def initialize_system():
    print("="*25)
    print("系统初始化中...")

    # === 1. 加载数据 ===
    start_time = time.time()
    print("[1/4] 正在加载用户和知识数据...")
    with open('data/users.json', 'r', encoding='utf-8') as file:
        user_prts = json.load(file)
    with open('data/knowledge.json', 'r', encoding='utf-8') as file:
        knowledge_prts = json.load(file)
    print(f"[1/4] 数据加载完毕，用时：{time.time() - start_time:.2f} 秒")

    # === 2. 加载模型 ===
    start_time = time.time()
    print("[2/4] 正在加载Bi-Encoder模型...")
    bi_encoder = SentenceTransformer('shibing624/text2vec-base-chinese', device='cpu')
    print(f"[2/4] 模型加载完毕，用时：{time.time() - start_time:.2f} 秒")

    # === 3. 准备知识库句子和嵌入 ===
    start_time = time.time()
    print("[3/4] 正在准备知识库句子和嵌入...")
    cache_sentences_path = os.path.join(CACHE_DIR, 'knowledge_sentences.json')
    cache_embeddings_path = os.path.join(CACHE_DIR, 'knowledge_embeddings.npy')

    if os.path.exists(cache_sentences_path) and os.path.exists(cache_embeddings_path):
        print("    > 检测到缓存，正在加载...")
        with open(cache_sentences_path, 'r', encoding='utf-8') as f:
            knowledge_sentences = json.load(f)
        knowledge_embeddings = np.load(cache_embeddings_path)
    else:
        print("    > 无缓存，开始生成句子和嵌入...")
        knowledge_sentences = [
            item.get('summary') or values_to_sentence(item)
            for item in knowledge_prts
        ]
        knowledge_embeddings = bi_encoder.encode(
            knowledge_sentences,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        with open(cache_sentences_path, 'w', encoding='utf-8') as f:
            json.dump(knowledge_sentences, f, ensure_ascii=False, indent=2)
        np.save(cache_embeddings_path, knowledge_embeddings)
        print("    > 缓存已保存。")
    print(f"[3/4] 知识库句子和嵌入准备完毕，用时：{time.time() - start_time:.2f} 秒")

    # === 4. 构建Faiss索引 ===
    start_time = time.time()
    print("[4/4] 正在构建Faiss索引...")
    dimension = bi_encoder.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(knowledge_embeddings)
    index.add(knowledge_embeddings)
    print(f"[4/4] Faiss索引构建完毕，用时：{time.time() - start_time:.2f} 秒")
    print("系统初始化完成。")
    print("="*25)

    return user_prts, knowledge_prts, bi_encoder, knowledge_sentences, knowledge_embeddings, index

def get_recommendation(user_id, user_prts, knowledge_prts, bi_encoder, knowledge_sentences, index):
    print("="*25)
    print(f"开始为用户 {user_id} 生成推荐...")

    # === 1. 准备用户画像句子 ===
    start_time = time.time()
    print("[1/3] 正在准备用户画像句子...")
    user_info_raw = values_to_sentence(user_prts[user_id])
    user_info_enhanced = user_prts[user_id].get('profile_summary') or user_info_raw
    print(f"[1/3] 用户画像句子准备完毕，用时：{time.time() - start_time:.2f} 秒")

    # === 2. 向量召回 ===
    start_time = time.time()
    print("[2/3] 正在进行向量召回...")
    user_embedding_np = bi_encoder.encode([user_info_enhanced], normalize_embeddings=True)
    faiss.normalize_L2(user_embedding_np)
    distances, indices = index.search(user_embedding_np, 20)
    print(f"[2/3] 召回完毕（共 {len(indices[0])} 条），用时：{time.time() - start_time:.2f} 秒")

    # === 3. LLM推荐分类 ===
    start_time = time.time()
    print("[3/3] 正在使用LLM生成推荐类别和内容...")
    retrieved_indices = indices[0]
    candidates_with_meta = [
        {
            "original_index": idx,
            "domain": knowledge_prts[idx].get("technical_domain", "未知领域"),
            "sentence": knowledge_sentences[idx]
        }
        for idx in retrieved_indices
    ]
    categorized_recommendations = llm_categorize_and_recommend(user_info_enhanced, candidates_with_meta)

    final_recommendations = {}
    for category, recommended_indices in categorized_recommendations.items():
        final_recommendations[category] = []
        for rec_idx in recommended_indices:
            original_knowledge_index = candidates_with_meta[rec_idx]["original_index"]
            final_recommendations[category].append(knowledge_prts[original_knowledge_index])

    print(f"[3/3] 推荐类别生成完毕，用时：{time.time() - start_time:.2f} 秒")
    print("="*25)

    return {
        "user_info": user_info_raw,
        "recommendations": final_recommendations
    }

def search_and_categorize_knowledge(query, knowledge_prts, bi_encoder, knowledge_sentences, index):
    print("="*25)
    print(f"开始为查询 '{query}' 进行知识搜索和分类...")

    # === 1. 编码查询 ===
    start_time = time.time()
    print("[1/3] 正在编码查询...")
    query_embedding_np = bi_encoder.encode([query], normalize_embeddings=True)
    faiss.normalize_L2(query_embedding_np)
    print(f"[1/3] 查询编码完毕，用时：{time.time() - start_time:.2f} 秒")

    # === 2. 向量召回 ===
    start_time = time.time()
    print("[2/3] 正在进行向量召回...")
    distances, indices = index.search(query_embedding_np, 20)
    print(f"[2/3] 召回完毕（共 {len(indices[0])} 条），用时：{time.time() - start_time:.2f} 秒")

    # === 3. LLM推荐分类 ===
    start_time = time.time()
    print("[3/3] 正在使用LLM生成推荐类别和内容...")
    retrieved_indices = indices[0]
    candidates_with_meta = [
        {
            "original_index": idx,
            "domain": knowledge_prts[idx].get("technical_domain", "未知领域"),
            "sentence": knowledge_sentences[idx]
        }
        for idx in retrieved_indices
    ]
    categorized_recommendations = llm_categorize_and_recommend(query, candidates_with_meta)

    final_recommendations = {}
    for category, recommended_indices in categorized_recommendations.items():
        final_recommendations[category] = []
        for rec_idx in recommended_indices:
            original_knowledge_index = candidates_with_meta[rec_idx]["original_index"]
            final_recommendations[category].append(knowledge_prts[original_knowledge_index])

    print(f"[3/3] 推荐类别生成完毕，用时：{time.time() - start_time:.2f} 秒")
    print("="*25)

    return {
        "query": query,
        "recommendations": final_recommendations
    }


if __name__ == "__main__":
    user_prts, knowledge_prts, bi_encoder, knowledge_sentences, knowledge_embeddings, index = initialize_system()

    # 示例：为用户生成推荐
    target_user_id = 1
    recommendation_result = get_recommendation(target_user_id, user_prts, knowledge_prts, bi_encoder, knowledge_sentences, index)

    print("\n--- 最终用户推荐结果 ---")
    print(f"用户画像: {recommendation_result['user_info']}")
    print("\n--- 为您推荐的知识类别 ---")
    if recommendation_result['recommendations']:
        for category, items in recommendation_result['recommendations'].items():
            print(f"\n【{category}】")
            for item in items:
                print(f"  - {item['title']} (ID: {item['resource_id']})")
    else:
        print("未能生成有效的推荐类别。")

    print("\n" + "="*50 + "\n")

    # 示例：根据用户输入进行搜索和分类
    user_query = "如何高效种植玉米并防治病虫害？"
    search_result = search_and_categorize_knowledge(user_query, knowledge_prts, bi_encoder, knowledge_sentences, index)

    print("\n--- 最终搜索和分类结果 ---")
    print(f"用户查询: {search_result['query']}")
    print("\n--- 为您整理的知识类别 ---")
    if search_result['recommendations']:
        for category, items in search_result['recommendations'].items():
            print(f"\n【{category}】")
            for item in items:
                print(f"  - {item['title']} (ID: {item['resource_id']})")
    else:
        print("未能根据查询生成有效的知识类别。")

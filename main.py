import json
import os
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

def get_recommendation(user_id):
    print("="*20)
    print(f"开始为用户 {user_id} 生成推荐...")

    # --- 1. 加载数据 ---
    print("[1/6] 正在加载用户和知识数据...")
    with open('data/users_processed.json', 'r', encoding='utf-8') as file:
        user_prts = json.load(file)
    with open('data/cleared_knowledge_processed.json', 'r', encoding='utf-8') as file:
        knowledge_prts = json.load(file)
    print("数据加载完毕。")

    # --- 2. 加载模型 ---
    print("[2/6] 正在加载Bi-Encoder模型...")
    # bi_encoder = SentenceTransformer('bert-base-nli-mean-tokens', device='cpu')4
    # bi_encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')
    # 中文任务推荐（强烈推荐）
    bi_encoder = SentenceTransformer('shibing624/text2vec-base-chinese', device='cpu')

    
    print("模型加载完毕。")

    # --- 3. 准备句子 ---
    print("[3/6] 正在准备用户画像和知识库句子...")
    user_info_raw = values_to_sentence(user_prts[user_id])
    user_info_enhanced = user_prts[user_id].get('profile_summary', user_info_raw)
    print(user_info_enhanced)

    cache_sentences_path = os.path.join(CACHE_DIR, 'knowledge_sentences.json')
    cache_embeddings_path = os.path.join(CACHE_DIR, 'knowledge_embeddings.npy')

    if os.path.exists(cache_sentences_path) and os.path.exists(cache_embeddings_path):
        print("    > 检测到缓存，正在加载...")
        with open(cache_sentences_path, 'r', encoding='utf-8') as f:
            knowledge_sentences = json.load(f)
        knowledge_embeddings = np.load(cache_embeddings_path)
    else:
        print("    > 无缓存，开始生成句子和嵌入...")
        knowledge_sentences = list_to_sentences(knowledge_prts)
        knowledge_embeddings = bi_encoder.encode(
            knowledge_sentences,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        # 保存缓存
        with open(cache_sentences_path, 'w', encoding='utf-8') as f:
            json.dump(knowledge_sentences, f, ensure_ascii=False, indent=2)
        np.save(cache_embeddings_path, knowledge_embeddings)
        print("    > 缓存已保存。")
    print("句子准备完毕。")

    # --- 4. 构建Faiss索引 ---
    print("[4/6] 正在构建Faiss索引...")
    dimension = bi_encoder.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(knowledge_embeddings)
    index.add(knowledge_embeddings)
    print("Faiss索引构建完毕。")

    # --- 5. 向量召回 ---
    print("[5/6] 正在进行向量召回...")
    user_embedding_np = bi_encoder.encode([user_info_enhanced], normalize_embeddings=True)
    faiss.normalize_L2(user_embedding_np)
    distances, indices = index.search(user_embedding_np, 20)
    print(f"召回了 {len(indices[0])} 个候选结果。")

    # --- 6. LLM推荐分类 ---
    print("[6/6] 正在使用LLM生成推荐类别和内容...")
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

    print("推荐类别生成完毕。")
    print("="*20)

    output = {
        "user_info": user_info_raw,
        "recommendations": final_recommendations
    }
    return output

if __name__ == "__main__":
    target_user_id = 1
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

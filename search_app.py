import json
import os
import time
import faiss
import numpy as np
from llm_api import llm_extend, llm_categorize_and_recommend
from sentence_transformers import SentenceTransformer

# 从 main.py 导入必要的函数和变量
from main import initialize_system, search_and_categorize_knowledge, values_to_sentence

if __name__ == "__main__":
    # 初始化系统，加载模型和数据
    user_prts, knowledge_prts, bi_encoder, knowledge_sentences, knowledge_embeddings, index = initialize_system()

    print("\n--- 知识搜索与分类应用 ---")
    while True:
        user_query = input("请输入您的查询（输入 'exit' 退出）：")
        if user_query.lower() == 'exit':
            break

        if not user_query.strip():
            print("查询不能为空，请重新输入。")
            continue

        # 调用搜索和分类函数
        search_result = search_and_categorize_knowledge(user_query, knowledge_prts, bi_encoder, knowledge_sentences, index)

        print("\n--- 为您整理的知识类别 ---")
        if search_result['recommendations']:
            for category, items in search_result['recommendations'].items():
                print(f"\n【{category}】")
                for item in items:
                    print(f"  - {item['title']} (ID: {item['resource_id']})")
        else:
            print("未能根据查询生成有效的知识类别。")
        print("\n" + "="*50 + "\n")

    print("应用已退出。")
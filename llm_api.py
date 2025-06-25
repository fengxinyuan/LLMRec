import json
from openai import OpenAI

client = OpenAI(api_key="sk-2b56a25ee39a456c9df233a02b73dc88", base_url="https://api.deepseek.com")

def llm_extend(content):
    prompt = f"""你是一位资深的农业技术顾问。请仔细分析以下原始用户画像，并生成一个增强版的摘要。

[原始画像]:
{content}

[你的任务]:
1.  **识别核心信息**: 提取用户的基本情况，如作物类型、规模、地区、技术水平。
2.  **推断潜在需求**: 基于用户的痛点（如成本高、劳动力老龄化）和期望（如绿色认证、品牌建设），推断出他们对哪些类型的技术或知识最感兴趣（例如：节约成本的技术、省力高效的机械、提升产品附加值的方法等）。
3.  **生成增强摘要**: 将以上分析整合成一段连贯、信息密集的文本。这段摘要应清晰地描绘出用户的核心诉求和技术偏好，以便于精准匹配知识资源。

请直接输出增强后的摘要文本。"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a senior agricultural technology consultant tasked with creating an enhanced, insightful summary of a user profile."},
            {"role": "user", "content": prompt},
        ],
        stream=False,
        temperature=0.0,
    )
    return response.choices[0].message.content

def llm_categorize_and_recommend(user_info, candidates_with_meta):
    candidate_text = ""
    for i, data in enumerate(candidates_with_meta):
        # 使用原始索引 i 作为标识符
        candidate_text += f"【{i}】(领域: {data['domain']}) {data['sentence']}\n"

    prompt = f"""你是一个顶级的农业知识推荐专家。你的任务是根据用户画像，从候选知识列表中，为用户识别并推荐3个最感兴趣的知识类别（技术领域），并为每个类别挑选2-3个最相关的具体知识点。

[用户画像]:
{user_info}

[候选知识列表]:
{candidate_text}

请遵循以下步骤和格式要求：
1.  **识别3个核心类别**: 分析用户画像和候选知识，找出用户最可能感兴趣的3个“技术领域”。
2.  **为每个类别挑选知识**: 从候选列表中，为每个确定的类别挑选2到3个最匹配的知识点。
3.  **输出JSON**: 你的输出必须是一个JSON对象。该对象的键是推荐的3个技术领域（类别），值是一个包含该类别下推荐知识的原始索引的列表。

例如，如果用户最关心“病虫害防治”和“机械化种植”，你的输出应该类似这样：
{{
  "病虫害防治": [5, 2],
  "机械化种植": [0, 8, 12],
  "智慧农业": [1]
}}
其中，数字（如5, 2, 0）是候选知识列表的原始索引。
请直接输出JSON对象，不要包含任何其他解释或注释。"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a top-tier agricultural knowledge recommendation expert. Your output must be a single, clean JSON object mapping recommended categories to a list of original candidate indices."},
            {"role": "user", "content": prompt},
        ],
        stream=False,
        temperature=0.0,
    )

    try:
        content = response.choices[0].message.content
        if content.startswith("```json"):
            content = content[7:-4].strip()
        data = json.loads(content)
        return data
    except (json.JSONDecodeError, AttributeError, KeyError):
        return {}


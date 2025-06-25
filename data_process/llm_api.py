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

def llm_format_and_enhance_user_profile(user_profile_json):
    prompt = f"""你是一位顶级的农业数据科学家。你的任务是接收一个JSON格式的用户画像，对其进行深度清洗、格式化、优化和增强，然后返回一个结构统一、信息丰富的JSON对象。

[原始用户画像]:
{user_profile_json}

[处理指令]:
1.  **保留并优化所有字段**: 遍历原始JSON中的每一个键值对。
2.  **格式化关键字段**:
    -   `annual_income`: 统一格式为字符串 "min-max" (单位：万)。例如，"12-15万" 变为 "12-15万元"。
    -   `learning_frequency`: 统一为 "X次/月" 的格式。例如，"每周1次" 变为 "4次/月"。
    -   `operation_scale`: 确保单位统一，并以数字形式表示。
    -   `technical_level`: 统一为 "初级", "中级", "高级" 中的一个。
    -   对其他文本字段，去除不必要的词语，使其更精炼。
3.  **新增派生字段**:
    -   `annual_income_range_lakh`: 根据 `annual_income` 创建一个新字段，值为一个包含两个整数的列表，代表万元为单位的收入范围，例如 `[12, 15]`。
    -   `learning_frequency_monthly`: 根据 `learning_frequency` 创建一个新字段，值为一个浮点数，代表月均学习次数，例如 `4.0`。
4.  **生成画像摘要**:
    -   全面分析整个优化后的用户画像。
    -   创建一个新字段 `profile_summary`，内容是一段自然语言摘要，精准概括用户的核心特征、主要需求和面临的挑战。
5.  **输出最终JSON**: 你的输出必须是单个、完整的JSON对象，包含所有经过优化和格式化的原始字段，以及新增的派生字段和摘要字段。

[示例输出格式]:
{{
  "user_id": "AG101",
  "name": "陈麦丰",
  "age": 43,
  "gender": "男",
  "location": "河北省石家庄市",
  "user_type": "小麦种植户",
  "operation_scale": 280,
  "main_crop": "小麦",
  "secondary_crops": ["玉米"],
  "farming_method": "半机械化",
  "experience_years": 18,
  "education": "中专",
  "technical_level": "中级",
  "annual_income": "12-15",
  "main_knowledge_needs": ["节水灌溉", "抗倒伏品种"],
  "secondary_knowledge_needs": ["土壤检测", "市场预测"],
  "challenges": ["水资源短缺", "极端天气"],
  "development_goals": ["提高水效", "订单农业"],
  "annual_income_range_lakh": [12, 15],
  "learning_frequency_monthly": 4.0,
  "profile_summary": "陈麦丰是一位来自河北石家庄的经验丰富的小麦种植户，经营280亩土地，技术水平中等。他当前面临水资源短缺和极端天气的挑战，核心需求是节水灌溉技术和抗倒伏品种，长期目标是实现订单农业以提高水资源利用效率。"
}}

请直接输出最终的、完整的JSON对象，不要包含任何其他解释或注释。"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a top-tier agricultural data scientist. Your output must be a single, complete, and enhanced JSON object representing the user profile, with all fields optimized and formatted."},
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
        if "user_id" in data and "profile_summary" in data and "annual_income_range_lakh" in data:
            return data
        return None
    except (json.JSONDecodeError, AttributeError, KeyError):
        return None

def llm_format_and_enhance_knowledge_resource(resource_json):
    prompt = f"""你是一位顶级的农业数据科学家和知识库编辑。你的任务是接收一个JSON格式的知识资源，对其进行严格的、全方位的格式化、标准化和内容增强，确保输出的JSON对象结构统一、内容精炼、信息丰富。

[原始知识资源]:
{resource_json}

[严格处理指令]:
1.  **保留所有键，优化所有值**: 必须保留原始JSON中的所有键。对每个键的值进行审查和优化。
2.  **严格标准化**:
    -   `title`: 改写为更精炼、信息密度更高的标题。
    -   `resource_type`: 必须从 ["技术标准", "数字工具", "培训课程", "物联网系统", "可视化工具", "技术方案", "行业报告"] 中选择一个最接近的。
    -   `knowledge_level`: 必须从 ["初级", "中级", "高级"] 中选择。
    -   `cost_level`: 必须从 ["无", "低", "中", "高", "订阅制", "商业解决方案"] 中选择。
    -   `description`: 改写为一段信息完整、语言精炼的描述。
    -   `tags`: 审查、去重并补充3-5个最核心、最相关的标签。
    -   对其他所有文本字段，都进行适当的精简和润色，统一为专业、书面的风格。
3.  **新增摘要字段**:
    -   `summary`: 创建一个新字段，内容是一段约100-150字的专业摘要，精准概括该资源的核心价值、技术原理、适用对象、预期效益和使用前提。
4.  **输出最终JSON**: 你的输出必须是单个、完整的JSON对象，包含所有经过严格优化和标准化的字段，以及新增的 `summary` 字段。

[示例输出格式]:
{{
  "resource_id": "RES001",
  "title": "小麦赤霉病AI预警与绿色防控决策系统",
  "resource_type": "智慧农业系统",
  "content_type": "交互式应用",
  "source": "国家农业信息化工程技术研究中心",
  "knowledge_level": "中级",
  "cost_level": "中高",
  "description": "本系统整合气象数据、田间传感器与AI模型，为小麦抽穗扬花期的赤霉病提供精准施药决策支持，旨在减少农药使用，实现增产增收。",
  "tags": ["智慧农业", "赤霉病", "AI预警", "精准植保", "小麦"],
  "summary": "这是一款面向规模化小麦种植户的智慧农业应用。它利用物联网和人工智能技术，实时监测和预测小麦赤霉病的发生风险，并提供科学的绿色防控方案。该系统能显著降低30%的农药使用量，帮助用户每亩增收200-400元，是实现精准植保、降本增效的有力工具。使用该系统需要具备一定的硬件基础和植保知识。"
}}

请直接输出最终的、完整的JSON对象，不要包含任何其他解释或注释。"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a top-tier agricultural knowledge base editor. Your output must be a single, complete, and enhanced JSON object representing the knowledge resource."},
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
        if "resource_id" in data and "summary" in data:
            return data
        return None
    except (json.JSONDecodeError, AttributeError, KeyError):
        return None

def llm_generate_html_for_knowledge(resource_json):
    prompt = f"""你是一个专业的农业知识内容生成器。你的任务是根据一个知识资源的JSON对象，提取其核心信息，并生成一段结构清晰、内容丰富、适合阅读的纯文本文章正文。

[知识资源JSON]:
{resource_json}

[文章正文生成要求]:
1.  **内容提取**: 综合利用 `title`, `summary`, `description`, `target_crops`, `technical_domain`, `applicable_regions`, `tags` 等字段中的信息。
2.  **结构组织**: 将提取的信息组织成连贯的段落。每个段落必须用 `<p>` 标签包裹。
3.  **语言风格**: 使用专业、简洁、易懂的语言。
4.  **HTML输出**: 只输出包含 `<p>` 标签的文章正文内容，不需要完整的HTML文档结构（如 `<html>`, `<head>`, `<body>` 等），也不需要其他额外信息。

请直接输出包含 `<p>` 标签的文章正文内容，不要包含任何其他解释或注释。"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a professional agricultural knowledge content generator. Your output must be a single HTML string containing only paragraph tags."},
            {"role": "user", "content": prompt},
        ],
        stream=False,
        temperature=0.0,
    )
    return response.choices[0].message.content

def llm_generate_new_knowledge_resources(existing_resources_summary_json, num_to_generate=5):
    prompt = f"""你是一位顶级的农业知识体系构建专家。你的任务是分析现有的知识库资源，并创造性地生成一批全新的、高质量的知识资源，以填补空白、丰富体系。

[现有资源概览]:
{existing_resources_summary_json}

[任务要求]:
1.  **分析与创新**: 分析现有资源的领域、作物和类型分布，识别出可以补充的领域（如“土壤改良”、“农产品品牌营销”、“智慧灌溉”、“设施农业”等）或新的资源形式（如“成本效益分析工具”、“市场趋势报告”、“政策解读直播”等）。
2.  **生成新资源**: 生成 {num_to_generate} 个全新的、与现有资源不重复的知识资源。
3.  **确保多样性**: 新生成的资源应在作物、技术领域、资源类型等方面尽可能多样化。
4.  **完整JSON输出**: 你的输出必须是一个JSON数组，其中包含 {num_to_generate} 个完整的新知识资源对象。每个对象都应包含`resource_id` (使用`NEW_RES_`前缀，并确保ID唯一), `title`, `resource_type`, `content_type`, `source`, `publish_date`, `target_crops`, `technical_domain`, `description`, `tags`等关键字段。

请直接输出包含 {num_to_generate} 个新资源对象的JSON数组，不要包含任何其他解释或注释。"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a top-tier agricultural knowledge architect. Your output must be a single JSON array containing new, diverse knowledge resources."},
            {"role": "user", "content": prompt},
        ],
        stream=False,
        temperature=0.7,
        max_tokens=8192,
    )
    try:
        content = response.choices[0].message.content
        if content.startswith("```json"):
            content = content[7:-4].strip()
        data = json.loads(content)
        if isinstance(data, list) and data:
            print(f"LLM成功生成了 {len(data)} 条新资源。")
            return data
        
        print(f"警告: LLM返回了空列表或无效的数据。")
        return []
    except (json.JSONDecodeError, AttributeError, KeyError) as e:
        print(f"错误: 解析LLM响应失败 - {e}")
        return []
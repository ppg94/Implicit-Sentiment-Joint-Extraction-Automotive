import pandas as pd
import json
import os
from openai import OpenAI
from tqdm import tqdm
import textwrap

client = OpenAI(
    api_key="",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
LLM_MODEL_NAME = "qwen-plus"
DATA_FILE_PATH = './maiteng_2020_TSI.xlsx'
TARGET_POLARITY = 0
NUM_CLUSTERS_TO_GENERATE = 6 

RESULTS_DIR = './results/'
ANALYSIS_NAME = '迈腾负面口碑分析' 


def load_and_filter_negative_data(file_path):
    df = pd.read_excel(file_path)
    negative_df = df[df['polarity'] < TARGET_POLARITY].copy()
    return negative_df

def discover_overall_issues(all_negative_comments):

    comments_str = "\n".join([f"- {comment}" for comment in all_negative_comments])

    prompt = f"""你是一位顶级的汽车行业分析师，擅长从海量用户反馈中归纳核心问题。
这里有关于“迈腾2020款TSI车型”的所有负面用户评论。请仔细阅读它们，然后识别并总结出用户抱怨的主要问题类别。

任务要求：
1. 归纳出大约 {NUM_CLUSTERS_TO_GENERATE} 个核心问题类别。
2. 每个类别都需要一个简洁且精准的名称（例如：“变速箱顿挫问题”、“车内异响问题”）。
3. 为每个类别提供一句简短的定义描述。
4. 严格以JSON格式输出一个列表，其中每个对象包含 "category_name" 和 "description" 两个键。

---
用户评论:
{comments_str}
---

你的分析结果 (严格遵循JSON格式):
"""


    completion = client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        top_p=0.9
    )
    response_text = completion.choices[0].message.content.strip()
    
    start_index = response_text.find('[')
    end_index = response_text.rfind(']') + 1
    if start_index != -1 and end_index != -1:
        json_str = response_text[start_index:end_index]
        discovered_clusters = json.loads(json_str)
        
        if isinstance(discovered_clusters, list) and all('category_name' in item and 'description' in item for item in discovered_clusters):
            print("✅ 类别发现成功！")
            return discovered_clusters

    return None


def classify_comment_with_llm(comment, dynamic_clusters):
    """
    【阶段二】使用动态生成的类别对单条评论进行分类。
    """
    cluster_definitions = "\n".join([f"- **{cluster['category_name']}**: {cluster['description']}" for cluster in dynamic_clusters])
    cluster_names = [cluster['category_name'] for cluster in dynamic_clusters]
    
    prompt = f"""你是一位专业的汽车行业评论分析师。你的任务是阅读以下关于“迈腾2020款TSI车型”的负面用户评论，并将其精准地归类到一个最合适的子类别中。

请从以下动态生成的类别中选择一个：
{cluster_definitions}
- **其他问题**: 如果评论不属于以上任何类别。

任务要求：
1.  仔细阅读用户评论。
2.  从上面的列表中选择最匹配的一个类别。
3.  严格按照指定格式输出，只返回类别名称。

---
用户评论: "{comment}"
---

最匹配的类别:"""
    cluster_names.append("其他问题") 
    completion = client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    response_text = completion.choices[0].message.content.strip()
    
    if response_text in cluster_names:
        return response_text
    else:
        for name in cluster_names:
            if name in response_text:
                return name
        return "其他问题"
            



def display_and_save_analysis_report(df_results, analysis_name, discovered_clusters):
    report_lines = []
    
    report_lines.append("="*70)
    report_lines.append(f" ✨ {analysis_name} ✨")
    report_lines.append("="*70)

    report_lines.append("\n【本次分析动态生成的类别定义】")
    for cluster in discovered_clusters:
        report_lines.append(f"  - {cluster['category_name']}: {cluster['description']}")

    report_lines.append("\n【各类别问题统计与占比】")
    cluster_counts = df_results['cluster_label'].value_counts().reset_index()
    cluster_counts.columns = ['类别', '数量']
    total_count = cluster_counts['数量'].sum()
    cluster_counts['占比'] = (cluster_counts['数量'] / total_count).map('{:.2%}'.format)
    report_lines.append(cluster_counts.to_string(index=False))

    report_lines.append("\n\n【各类别典型用户评论示例】")
    for category in cluster_counts['类别']:
        count = cluster_counts.loc[cluster_counts['类别']==category, '数量'].iloc[0]
        report_lines.append(f"\n--- {category} (共{count}条) ---")
        
        sample_comments = df_results[df_results['cluster_label'] == category]['comment'].sample(
            n=min(3, count), random_state=42
        ).tolist()
        
        for i, comment in enumerate(sample_comments):
            wrapped_comment = textwrap.fill(comment, width=80, initial_indent=f"  [{i+1}] ", subsequent_indent="      ")
            report_lines.append(wrapped_comment)
            
    full_report_str = "\n".join(report_lines)
    print("\n" + full_report_str)

    try:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        file_name = f"{analysis_name.replace(' ', '_')}_report.txt"
        file_path = os.path.join(RESULTS_DIR, file_name)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(full_report_str)
            
        print("\n" + "="*70)
        print(f"✅ [成功] 分析报告已保存至: {file_path}")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ [错误] 保存报告文件失败: {e}")


if __name__ == "__main__":
    df_negative = load_and_filter_negative_data(DATA_FILE_PATH)
    
    if df_negative is not None and not df_negative.empty:
        all_negative_comments = df_negative['comment'].tolist()
        dynamic_clusters = discover_overall_issues(all_negative_comments)
        
        if dynamic_clusters:
            print(f"\n--- 阶段二：规模化分类 ---")
            print(f"正在使用新发现的类别对全部 {len(df_negative)} 条负面评论进行分类...")
            
            tqdm.pandas(desc="分类进度")
            df_negative['cluster_label'] = df_negative['comment'].progress_apply(
                classify_comment_with_llm, args=(dynamic_clusters,)
            )
            display_and_save_analysis_report(df_negative, ANALYSIS_NAME, dynamic_clusters)
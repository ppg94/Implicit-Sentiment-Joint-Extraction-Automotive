import pandas as pd
import numpy as np
import time
import os
import json
from openai import OpenAI
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import re


client = OpenAI(
    api_key="",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
LLM_MODEL_NAME = "qwen-plus"
DATA_FILE_PATH = './test.xlsx'
LEADERBOARD_CSV_PATH = 'product_issue_leaderboard.csv'
LLM_RESPONSES_DIR = './llm_responses/'
FEATURES_LIST_PATH = './features_list.json'
RESULTS_DIR = './results/'
EXPERIMENT_NAME = 'scored_enhance'
NUM_SPLITS = 10
RANDOM_STATE = 42
FEW_SHOT_POSITIVE_SAMPLES = 2
FEW_SHOT_NEGATIVE_SAMPLES = 3




def process_features_to_set(feature_string):
    if not isinstance(feature_string, str):
        return set()
    features = re.split(r'[,，、\s]+', feature_string)
    return set(f.strip() for f in features if f.strip())

def load_and_prepare_data(file_path):

    df = pd.read_excel(file_path)
    df_clean = df[['comment', 'feature', 'polarity', 'featureNum']].copy()
    
    df_binary = df_clean[df_clean['polarity'] != 0].copy()
    df_binary['sentiment_label'] = df_binary['polarity'].apply(lambda x: "正面" if x > 0 else "负面")
    df_binary['numeric_label'] = df_binary['polarity'].apply(lambda x: 1 if x > 0 else 0)
    
    df_binary = df_binary.reset_index().rename(columns={'index': 'original_index'})
    return df_binary


def save_llm_response(log_file_path, sample_index, response_text):
    with open(log_file_path, 'a', encoding='utf-8') as f:
        f.write(f"--- Sample Index: {sample_index} ---\n")
        f.write(response_text + "\n\n")

def create_joint_selection_prompt_v4(text_to_analyze, few_shot_examples, feature_list_str):
    examples_str = ""
    for ex in few_shot_examples:
        output_json_str = json.dumps({"feature": ex['feature'], "sentiment": ex['sentiment']}, ensure_ascii=False)
        examples_str += f"示例评论: \"{ex['comment']}\"\n分析结果: {output_json_str}\n\n"

    return f"""你是一个极其严谨的文本信息分类机器人。你的唯一任务是作为数据处理流程中的一个自动化节点，
严格按照指令，从给定的“评论”文本中，判断其核心讨论点，并从一个固定的列表中选择最相关的“产品特征”(feature)，同时判断其“情感倾向”(sentiment)。

你的输出将被Python脚本自动解析，任何格式错误都将导致流程失败。

【核心指令：选择而非提取】
'feature' 的值 **必须** 是从下面的【候选特征列表】中**选择**的最贴切的一项。
严禁进行任何形式的总结、归纳、或创造新词。你的任务是 **匹配** 和 **选择**，而不是从评论原文中提取。

【候选特征列表】
{feature_list_str}

【输出规则】
1.  **JSON格式: 最终输出必须是一个、且仅有一个有效的JSON对象。禁止包含任何解释、前缀、后缀、或Markdown的```json```标记。
2.  **内容键: JSON对象必须包含且仅包含 "feature" 和 "sentiment" 两个键。
3.  **单一情感: 'sentiment' 的值必须严格为 "正面" 或 "负面" 这两个字符串之一。
4.  **列表原则: 再次强调，'feature' 的值必须是【候选特征列表】中给出的一个。

**【学习示例】**
{examples_str}
---
**【开始分析】**
评论: "{text_to_analyze}"
分析结果: """

# 在这里，将原来的函数定义替换为这个新的
def get_prediction_from_llm(comment, few_shot_examples, feature_list_str, sample_index, log_file_path):
    prompt = create_joint_selection_prompt_v4(comment, few_shot_examples, feature_list_str)
    max_retries = 3
    for i in range(max_retries):
        completion = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        response_text = completion.choices[0].message.content.strip()
        save_llm_response(log_file_path, sample_index, response_text)
        
        start_index = response_text.find('{')
        end_index = response_text.rfind('}') + 1
        if start_index != -1 and end_index != -1:
            json_str = response_text[start_index:end_index]
            parsed_json = json.loads(json_str)
            if 'feature' in parsed_json and 'sentiment' in parsed_json:
                return parsed_json['feature'], parsed_json['sentiment']
            
    return None, None

def generate_and_save_report(df, report_df, sentiment_accuracy, feature_accuracy, end_to_end_accuracy, experiment_name):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_header = (
        f"\n{'='*50}\n"
        f" 实验名称: {experiment_name}\n"
        f"{'='*50}\n"
        f"总测试样本数: {len(df)}\n"
        f"成功解析样本数: {len(report_df)}\n"
        f"{'-'*50}\n"
        f"情感分类准确率 (Sentiment Accuracy): {sentiment_accuracy:.4f}\n"
        f"特征选择准确率 (Feature Accuracy - 灵活匹配): {feature_accuracy:.4f}\n"
        f"端到端准确率 (End-to-End Accuracy - 灵活匹配): {end_to_end_accuracy:.4f}\n"
        f"{'-'*50}\n"
    )
    
    classification_rep_str = classification_report(report_df['true_sentiment_label'], report_df['pred_sentiment'])
    
    full_report_str = report_header + "情感分类详细报告:\n" + classification_rep_str

    report_file_path = os.path.join(RESULTS_DIR, f"{experiment_name}.txt")
    with open(report_file_path, 'w', encoding='utf-8') as f:
        f.write(full_report_str)



    print(full_report_str)


def run_cross_validation(df, feature_list):
    os.makedirs(LLM_RESPONSES_DIR, exist_ok=True)
    feature_list_for_prompt = ", ".join(feature_list)
    X = df['comment']
    y = df['numeric_label']
    indices = df.index.values

    skf = StratifiedKFold(n_splits=NUM_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    all_results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"第 {fold}/{NUM_SPLITS} 折")
        log_file_path = os.path.join(LLM_RESPONSES_DIR, f"fold_{fold}_responses.txt")
        if os.path.exists(log_file_path):
            os.remove(log_file_path)

        train_df_fold = df.iloc[train_idx]
        test_df_fold = df.iloc[test_idx]

        pos_samples = train_df_fold[train_df_fold['numeric_label'] == 1].sample(FEW_SHOT_POSITIVE_SAMPLES, random_state=fold)
        neg_samples = train_df_fold[train_df_fold['numeric_label'] == 0].sample(FEW_SHOT_NEGATIVE_SAMPLES, random_state=fold)
        
        few_shot_examples = []
        for _, row in pd.concat([neg_samples, pos_samples]).iterrows():
            few_shot_examples.append({
                "comment": row['comment'],
                "feature": row['feature'],
                "sentiment": row['sentiment_label']
            })

        for row in test_df_fold.itertuples():
            pred_feature, pred_sentiment = get_prediction_from_llm(
                row.comment, few_shot_examples, feature_list_for_prompt, row.Index, log_file_path
            )
            # 在结果中加入 featureNum
            all_results.append({
                'comment': row.comment, 'true_feature': row.feature,
                'true_sentiment_label': row.sentiment_label,
                'pred_feature': pred_feature, 'pred_sentiment': pred_sentiment,
                'featureNum': row.featureNum # <--- 新增
            })
            
    results_df = pd.DataFrame(all_results).dropna(subset=['pred_feature', 'pred_sentiment'])
    report_df = results_df[results_df['pred_sentiment'].isin(['负面', '正面'])].copy()


    feature_score_total = 0.0
    end_to_end_score_total = 0.0

    for _, row in report_df.iterrows():
        true_features_set = process_features_to_set(row['true_feature'])
        pred_features_set = process_features_to_set(row['pred_feature'])
        
        num_true_features = len(true_features_set)
        feature_score = 0.0
        
        if num_true_features > 0:
            common_features_count = len(true_features_set.intersection(pred_features_set))
            
            feature_score = common_features_count / num_true_features
            feature_score_total += feature_score
        
        if row['true_sentiment_label'] == row['pred_sentiment']:
            end_to_end_score_total += feature_score

    total_samples = len(report_df)
    
    sentiment_accuracy = accuracy_score(report_df['true_sentiment_label'], report_df['pred_sentiment'])
    
    feature_accuracy_flexible = feature_score_total / total_samples if total_samples > 0 else 0
    
    end_to_end_accuracy_flexible = end_to_end_score_total / total_samples if total_samples > 0 else 0

    generate_and_save_report(
        df, 
        report_df, 
        sentiment_accuracy, 
        feature_accuracy_flexible, 
        end_to_end_accuracy_flexible,
        EXPERIMENT_NAME  
    )
    
    return results_df

def generate_business_insights(df_with_predictions):
    
    df_filtered = df_with_predictions[df_with_predictions['pred_sentiment'].isin(['正面', '负面'])].copy()
    
    insight_summary = df_filtered.groupby('pred_feature')['pred_sentiment'].value_counts().unstack(fill_value=0)
    if '正面' not in insight_summary: insight_summary['正面'] = 0
    if '负面' not in insight_summary: insight_summary['负面'] = 0
        
    insight_summary['总计'] = insight_summary['正面'] + insight_summary['负面']
    insight_summary['负面比例'] = (insight_summary['负面'] / insight_summary['总计']).round(2)
    
    top_complaints = insight_summary.sort_values(by='负面', ascending=False)
    
    print(top_complaints.head(10))
    top_complaints.to_csv(LEADERBOARD_CSV_PATH, encoding='utf-8-sig')
    print(f"产品问题排行榜: {LEADERBOARD_CSV_PATH}")

    if not top_complaints.empty:
        most_complained_feature = top_complaints.index[0]
        specific_complaints = df_with_predictions[
            (df_with_predictions['pred_feature'] == most_complained_feature) &
            (df_with_predictions['pred_sentiment'] == '负面')
        ]['comment']
        
        sample_size = min(5, len(specific_complaints))
        for comment in specific_complaints.sample(sample_size, random_state=RANDOM_STATE).tolist():
            print(f"- “{comment}”")


def load_feature_list(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        features = json.load(f)
        if isinstance(features, list) and all(isinstance(item, str) for item in features):
            return features


if __name__ == "__main__":
    feature_list = load_feature_list(FEATURES_LIST_PATH)
    if feature_list is None:
        exit()
    feature_list = load_feature_list(FEATURES_LIST_PATH)
    if feature_list is None:
        exit()
    
    df_original = load_and_prepare_data(DATA_FILE_PATH)
    if df_original is not None:
        df_cv_results = run_cross_validation(df_original, feature_list) 
        if not df_cv_results.empty:
            generate_business_insights(df_cv_results)
import pandas as pd
import re
import json
from collections import Counter


TXT_FILE_PATH = ''
XLSX_FILE_PATH = './test.xlsx'



def parse_llm_responses(file_path):
    responses = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            pattern = re.compile(r'--- Sample Index: (\d+) ---\s*({.*?})', re.DOTALL)
            matches = pattern.findall(content)
            
            for match in matches:
                index = int(match[0])
                data = json.loads(match[1])
                responses[index] = data
    except FileNotFoundError:
        print(f"Error:Can't find {file_path}")
        return None
    return responses

def map_sentiment_to_polarity(sentiment_str):

    mapping = {
        '正面': 1,
        '中性': 0,
        '负面': -1
    }
    return mapping.get(sentiment_str, None)

def process_features_to_set(feature_string):

    if not isinstance(feature_string, str):
        return set()

    features = re.split(r'[,，、\s]+', feature_string)

    return set(f.strip() for f in features if f.strip())

def evaluate_matches():

    llm_data = parse_llm_responses(TXT_FILE_PATH)
    if llm_data is None:
        return

    try:
        ground_truth_df = pd.read_excel(XLSX_FILE_PATH)
    except FileNotFoundError:
        print(f"错误: 找不到Excel文件 {XLSX_FILE_PATH}")
        return
    except Exception as e:
        print(f"读取Excel文件时发生错误: {e}")
        return


    total_samples = len(llm_data)
    feature_matches = 0
    sentiment_matches = 0
    combined_matches = 0 
    
    print("--- 开始评估 ---")
    mismatched_samples = []

    for index, llm_response in llm_data.items():
        df_index = index
        
        if df_index >= len(ground_truth_df):
            print(f"警告: Sample Index {index} 对应的行 {df_index+1} 超出Excel文件范围，跳过。")
            continue

        ground_truth_row = ground_truth_df.iloc[df_index]
        gt_feature_str = ground_truth_row['feature']
        gt_feature_num = ground_truth_row['featureNum']
        gt_polarity = ground_truth_row['polarity']

        llm_feature_str = llm_response.get('feature', '')
        llm_sentiment_str = llm_response.get('sentiment', '')

        gt_features_set = process_features_to_set(gt_feature_str)
        llm_features_set = process_features_to_set(llm_feature_str)
        llm_polarity = map_sentiment_to_polarity(llm_sentiment_str)
        
        is_feature_match = False
        common_features_count = len(gt_features_set.intersection(llm_features_set))

        if gt_feature_num == 1:
            if gt_features_set.issubset(llm_features_set):
                is_feature_match = True
        elif gt_feature_num == 2:
            if common_features_count >= 1:
                is_feature_match = True
        elif gt_feature_num == 3:
            if common_features_count >= 2:
                is_feature_match = True
        elif gt_feature_num == 4:
            if common_features_count >= 3:
                is_feature_match = True
        
        is_sentiment_match = (gt_polarity == llm_polarity)

        if is_feature_match:
            feature_matches += 1
        if is_sentiment_match:
            sentiment_matches += 1
        if is_feature_match and is_sentiment_match:
            combined_matches += 1
        
        if not (is_feature_match and is_sentiment_match):
            mismatched_samples.append({
                'Sample Index': index,
                'Excel Row': df_index + 2,
                'GT Feature': gt_feature_str,
                'LLM Feature': llm_feature_str,
                'Feature Match': is_feature_match,
                'GT Sentiment': gt_polarity,
                'LLM Sentiment': f"{llm_sentiment_str}({llm_polarity})",
                'Sentiment Match': is_sentiment_match,
            })

    print("\n--- 评估结果摘要 ---")
    print(f"总共处理样本数: {total_samples}")
    print("-" * 20)
    
    if total_samples > 0:
        feature_accuracy = (feature_matches / total_samples) * 100
        sentiment_accuracy = (sentiment_matches / total_samples) * 100
        combined_accuracy = (combined_matches / total_samples) * 100
        
        print(f"特征匹配成功数: {feature_matches} / {total_samples}")
        print(f"特征匹配准确率: {feature_accuracy:.2f}%")
        print("-" * 20)
        
        print(f"情感匹配成功数: {sentiment_matches} / {total_samples}")
        print(f"情感匹配准确率: {sentiment_accuracy:.2f}%")
        print("-" * 20)
        
        print(f"特征和情感均匹配成功数: {combined_matches} / {total_samples}")
        print(f"综合准确率: {combined_accuracy:.2f}%")
    else:
        print("没有可处理的样本。")

    print("\n--- 部分不匹配样本详情 (最多显示10条) ---")
    if not mismatched_samples:
        print("所有样本均完全匹配！")
    else:
        for i, sample in enumerate(mismatched_samples):
            print(f"\n[不匹配样本 {i+1}]")
            for key, value in sample.items():
                print(f"  {key}: {value}")

if __name__ == "__main__":
    evaluate_matches()
from openai import OpenAI
import os
import time
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, classification_report

data_path = './test.xlsx' 

LLM_MODEL_NAME = "qwen2-7b-instruct" 


df = pd.read_excel(data_path)
df_clean = df[['comment', 'polarity']].copy()
df_binary = df_clean.copy()
df_binary['label'] = df_binary['polarity'].apply(lambda x: 1 if x > 0 else 0)
X = df_binary['comment']
y = df_binary['label']


client = OpenAI(
    api_key="",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)



def create_knowledge_prompt(text_to_classify, few_shot_examples):
    examples_str = ""
    for ex in few_shot_examples:
        label_str = "正面" if ex['label'] == 1 else "负面"
        examples_str += f"示例评论: \"{ex['comment']}\"\n情感倾向: {label_str}\n\n"
        
#     prompt = f"""你是一位资深的汽车行业评论分析专家。你的任务是分析用户评论中隐藏的、间接表达的情感倾向。
# 在汽车领域，“油耗高”、“悬挂硬”、“异响”通常是负面评价，而“推背感强”、“空间大”、“指哪打哪”通常是正面评价。
    prompt = f"""请严格按照下面的示例进行判断，只回答“正面”或“负面”。
---
{examples_str}
---

现在，请判断以下评论的情感倾向：

评论: "{text_to_classify}"
情感倾向: """
    return prompt


method_1_scores = []
all_true_labels_m1 = []
all_predicted_labels_m1 = []

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(f"第 {fold+1}/10 折")
    
    train_df_fold = df_binary.iloc[train_index]
    test_df_fold = df_binary.iloc[test_index]
    
    neg_samples = train_df_fold[train_df_fold['label'] == 0].sample(3, random_state=fold)
    pos_samples = train_df_fold[train_df_fold['label'] == 1].sample(2, random_state=fold)
    few_shot_examples = pd.concat([neg_samples, pos_samples]).to_dict('records')


    test_comments = test_df_fold['comment'].tolist()
    true_labels = test_df_fold['label'].tolist()
    predicted_labels = []
    
    for i, comment in enumerate(test_comments):
        prompt = create_knowledge_prompt(comment, few_shot_examples)

        completion = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            extra_body={"enable_thinking": False} 
        )
        response_text = completion.choices[0].message.content.strip()
        
        predicted_label = 1 if "正面" in response_text else 0
        predicted_labels.append(predicted_label)
        
        print(f"Fold {fold+1}, Sample {i+1}/{len(test_comments)}: Predicted '{response_text}' -> {predicted_label}")
            

    valid_indices = [i for i, label in enumerate(predicted_labels) if label != -1]
    valid_true = [true_labels[i] for i in valid_indices]
    valid_pred = [predicted_labels[i] for i in valid_indices]

    acc = accuracy_score(valid_true, valid_pred) if len(valid_true) > 0 else 0
    method_1_scores.append(acc)
    all_true_labels_m1.extend(valid_true)
    all_predicted_labels_m1.extend(valid_pred)
    
    print(f"第 {fold+1} 折的准确率: {acc:.4f}")

mean_accuracy = np.mean(method_1_scores)
print(f"平均准确率: {mean_accuracy:.4f}")

print(classification_report(all_true_labels_m1, all_predicted_labels_m1, target_names=['负面(0)', '正面(1)']))
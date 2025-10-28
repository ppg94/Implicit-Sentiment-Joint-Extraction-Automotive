import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, classification_report
import torch


data_path = './ei_paper/test.xlsx' 
df = pd.read_excel(data_path)
df_clean = df[['comment', 'polarity']].copy()
df_binary = df_clean.copy()

df_binary['label'] = df_binary['polarity'].apply(lambda x: 1 if x > 0 else 0)

X = df_binary['comment']
y = df_binary['label']


skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

local_model_path = './ei_paper/hlf/'

device = 0 if torch.cuda.is_available() else -1
model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
text_classification = pipeline(
    'sentiment-analysis', 
    model=model, 
    tokenizer=tokenizer, 
    device=device 
)

baseline_a_accuracies = []
all_true_labels = []
all_predicted_labels = []

for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(f"第 {fold+1} 折")

    X_test_fold = X.iloc[test_index].tolist()
    y_test_fold = y.iloc[test_index].tolist()
    

    predictions = text_classification(X_test_fold, batch_size=16, truncation=True)

    if fold == 0:
        print(f"模型输出示例: {predictions[0]}")

    positive_model_label = 'positive' 
    label_map = {positive_model_label: 1} 
    
    predicted_labels = [label_map.get(p['label'].lower(), 0) for p in predictions]

    acc = accuracy_score(y_test_fold, predicted_labels)
    baseline_a_accuracies.append(acc)
    
    all_true_labels.extend(y_test_fold)
    all_predicted_labels.extend(predicted_labels)
    
    print(f"第 {fold+1} 折的准确率: {acc:.4f}")


mean_accuracy = np.mean(baseline_a_accuracies)
print(f"平均准确率: {mean_accuracy:.4f}")

print(classification_report(all_true_labels, all_predicted_labels, target_names=['负面(0)', '正面(1)']))
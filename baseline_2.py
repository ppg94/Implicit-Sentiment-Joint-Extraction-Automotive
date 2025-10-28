import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from datasets import Dataset
from sklearn.metrics import classification_report
import torch

data_path = './test.xlsx' 

df = pd.read_excel(data_path)

df_clean = df[['comment', 'polarity']].copy()

df_binary = df_clean.copy()

df_binary['label'] = df_binary['polarity'].apply(lambda x: 1 if x > 0 else 0)
X = df_binary['comment']
y = df_binary['label']

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


local_model_path = './bge-large-zh-v1.5/'

tokenizer = AutoTokenizer.from_pretrained(local_model_path)


def preprocess_function(examples):
    return tokenizer(examples['comment'], truncation=True, max_length=128)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    report = classification_report(labels, predictions, output_dict=True, zero_division=0)
    return {
        'accuracy': report['accuracy'],
        'f1': report['weighted avg']['f1-score'],
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall']
    }


baseline_b_scores = []
all_true_labels_b = []
all_predicted_labels_b = []


for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(f"第 {fold+1} 折")
    
    torch.cuda.empty_cache()
    
    train_df_fold = df_binary.iloc[train_index]
    test_df_fold = df_binary.iloc[test_index]
    
    train_dataset = Dataset.from_pandas(train_df_fold)
    test_dataset = Dataset.from_pandas(test_df_fold)

    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)
    
    if '__index_level_0__' in tokenized_train_dataset.column_names:
        tokenized_train_dataset = tokenized_train_dataset.remove_columns(['comment', 'polarity', '__index_level_0__'])
        tokenized_test_dataset = tokenized_test_dataset.remove_columns(['comment', 'polarity', '__index_level_0__'])
    else:
        tokenized_train_dataset = tokenized_train_dataset.remove_columns(['comment', 'polarity'])
        tokenized_test_dataset = tokenized_test_dataset.remove_columns(['comment', 'polarity'])


    model = AutoModelForSequenceClassification.from_pretrained(local_model_path, num_labels=2)

    training_args = TrainingArguments(
        output_dir=f'./results_fold_{fold+1}',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f'./logs_fold_{fold+1}',
        logging_steps=50, 
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",  
        report_to="none" 
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator, 
    )

    trainer.train()

    predictions_output = trainer.predict(tokenized_test_dataset)
    y_true_fold = predictions_output.label_ids
    y_pred_fold = np.argmax(predictions_output.predictions, axis=1)
    
    acc = accuracy_score(y_true_fold, y_pred_fold)
    baseline_b_scores.append(acc)
    print(f"第 {fold+1} 折的最佳准确率: {acc:.4f}")
    
    all_true_labels_b.extend(y_true_fold)
    all_predicted_labels_b.extend(y_pred_fold)
    
    del model
    del trainer
    torch.cuda.empty_cache()

mean_accuracy = np.mean(baseline_b_scores)
print(f"平均准确率: {mean_accuracy:.4f}")

print(classification_report(all_true_labels_b, all_predicted_labels_b, target_names=['负面(0)', '正面(1)']))
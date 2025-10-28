import pandas as pd
import json

def create_feature_list(excel_path, output_path='features_list.json'):

    df = pd.read_excel(excel_path)

    if 'feature' not in df.columns:
        print("No col named 'feature'")
        return
    unique_features_set = set()
    
    for item in df['feature'].dropna():
        features = [feature.strip() for feature in str(item).split(',')]
        unique_features_set.update(features)

    unique_features_list = sorted(list(unique_features_set))
    
    print(f"All {len(unique_features_list)} fratures are found")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(unique_features_list, f, ensure_ascii=False, indent=4)
    
    print(f"Result has saved in {output_path}")


if __name__ == "__main__":
    file_path = './test.xlsx'
    create_feature_list(file_path, output_path='features_list.json')
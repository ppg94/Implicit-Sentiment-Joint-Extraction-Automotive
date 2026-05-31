# Pattern-Guided Implicit Sentiment Joint Extraction for Automotive User Feedback

![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Task](https://img.shields.io/badge/task-implicit%20sentiment%20analysis-green.svg)
![Domain](https://img.shields.io/badge/domain-automotive%20reviews-orange.svg)

This repository contains the code and sample data for **Pattern-Guided Implicit Sentiment Analysis**, a knowledge-augmented prompting pipeline for automotive user feedback. The project focuses on reviews where user sentiment is often implied through domain-specific expressions such as vehicle components, driving experience, defects, configuration, or maintenance-related terms.

The repository supports three related tasks:

- Binary sentiment classification for automotive reviews.
- Aspect-sentiment joint extraction, where each review is mapped to a standardized automotive feature and a sentiment label.
- Negative feedback clustering, where LLMs discover and summarize recurring product issue categories.

## Paper

This code accompanies the SAE Technical Paper:

**Pattern-Guided Implicit Sentiment Analysis: Knowledge-Augmented Prompting with Aspect-Sentiment Joint Extraction for Automotive User Feedback**

Gengjia Chang, Zuxing Deng, Aonan Ma, Jiangqi Yao, Xiaojian Li, and Ling Li

SAE Technical Paper 2026-99-0753, 2026

DOI: [10.4271/2026-99-0753](https://doi.org/10.4271/2026-99-0753)

The paper reports experiments on a corpus of 2,062 Chinese automotive feedback samples and a case study on Volkswagen Magotan negative reviews.

## Repository Structure

```text
.
├── api.py                         # LLM few-shot binary sentiment classification
├── baseline_1.py                  # Local model zero-shot sentiment baseline
├── baseline_2.py                  # Local model fine-tuning sentiment baseline
├── feature_extract2.py            # Main aspect-sentiment joint extraction pipeline
├── feature_extract2_noCoT.py      # No-CoT variant for joint extraction experiments
├── clustering.py                  # Two-stage negative feedback discovery and clustering
├── ensure.py                      # Evaluation utility for saved LLM JSON responses
├── get_dict.py                    # Builds features_list.json from the annotated dataset
├── features_list.json             # Standardized automotive feature vocabulary
├── test.xlsx                      # Main annotated review dataset
├── maiteng_2020_TSI.xlsx          # Magotan case-study dataset
└── product_issue_leaderboard.csv  # Example aggregated issue leaderboard
```

## Data Format

The scripts expect Excel files with the following columns:

| Column | Description |
| --- | --- |
| `comment` | Raw user review text. |
| `feature` | Ground-truth automotive feature or aspect. Multiple features may be comma-separated. |
| `featureNum` | Number of annotated features in `feature`. |
| `polarity` | Sentiment label. Positive values are positive, negative values are negative, and `0` is neutral. |

`test.xlsx` contains 2,062 annotated samples. `maiteng_2020_TSI.xlsx` is used by `clustering.py` for the Magotan case study.

## Installation

Create an environment and install the Python dependencies:

```bash
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

python -m pip install -U pip
python -m pip install pandas numpy scikit-learn torch transformers datasets openai tqdm openpyxl
```

The LLM scripts use an OpenAI-compatible API client. By default, they point to the DashScope compatible endpoint:

```python
client = OpenAI(
    api_key="",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
```

Before running LLM-based experiments, set your API key in the corresponding script. Do not commit private keys to the repository.

## Quick Start

### 1. Generate or refresh the feature vocabulary

`feature_extract2.py` uses a fixed candidate feature list. To rebuild it from `test.xlsx`, run:

```bash
python get_dict.py
```

This writes `features_list.json`.

### 2. Run LLM few-shot sentiment classification

```bash
python api.py
```

This script runs 10-fold stratified cross-validation over `test.xlsx`, samples few-shot examples from each training fold, and reports accuracy plus a classification report.

### 3. Run local sentiment baselines

```bash
python baseline_1.py
python baseline_2.py
```

Update the local model and data paths inside the scripts before running:

- `baseline_1.py`: `data_path`, `local_model_path`
- `baseline_2.py`: `data_path`, `local_model_path`

`baseline_1.py` evaluates a local classifier without fine-tuning. `baseline_2.py` fine-tunes a local Transformer model under 10-fold cross-validation.

### 4. Run aspect-sentiment joint extraction

```bash
python feature_extract2.py
```

The pipeline performs feature selection and sentiment classification in one JSON-formatted LLM response. It uses:

- `test.xlsx` as the annotated dataset.
- `features_list.json` as the standardized candidate feature list.
- 10-fold stratified cross-validation.
- Few-shot prompting with positive and negative examples from each training fold.

Main outputs:

- `llm_responses/`: raw LLM responses per fold.
- `results/scored_enhance.txt`: sentiment, feature, and end-to-end accuracy report.
- `product_issue_leaderboard.csv`: aggregated feature-level positive/negative counts.

To run the no-CoT variant:

```bash
python feature_extract2_noCoT.py
```

### 5. Run negative feedback clustering

```bash
python clustering.py
```

`clustering.py` uses `maiteng_2020_TSI.xlsx` and performs a two-stage workflow:

1. The LLM reads all negative comments and discovers major issue categories.
2. Each negative comment is classified into one discovered category or `其他问题`.

The report is saved under `results/`.

## Example Output

`product_issue_leaderboard.csv` aggregates predicted feature-sentiment pairs into issue-level counts:

| pred_feature | 正面 | 负面 | 总计 | 负面比例 |
| --- | ---: | ---: | ---: | ---: |
| 油耗 | 69 | 76 | 145 | 0.52 |
| 发动机 | 6 | 42 | 48 | 0.88 |
| 后备箱 | 8 | 38 | 46 | 0.83 |
| 配置 | 16 | 36 | 52 | 0.69 |
| 刹车 | 1 | 31 | 32 | 0.97 |

This output can be used to identify high-frequency negative product issues and prioritize quality improvement or market analysis.

## Evaluation Notes

The joint extraction scripts report three metrics:

- **Sentiment Accuracy**: whether the predicted sentiment matches the ground-truth polarity.
- **Feature Accuracy**: flexible feature matching between predicted and annotated feature sets.
- **End-to-End Accuracy**: feature match weighted by correct sentiment prediction.

`ensure.py` can evaluate saved LLM response logs. Set `TXT_FILE_PATH` to the target response file and run:

```bash
python ensure.py
```

## Citation

If this repository helps your research, please cite:

```bibtex
@techreport{Chang2026PatternGuidedIS,
  title = {Pattern-Guided Implicit Sentiment Analysis: Knowledge-Augmented Prompting with Aspect-Sentiment Joint Extraction for Automotive User Feedback},
  author = {Chang, Gengjia and Deng, Zuxing and Ma, Aonan and Yao, Jiangqi and Li, Xiaojian and Li, Ling},
  institution = {SAE International},
  number = {2026-99-0753},
  year = {2026},
  doi = {10.4271/2026-99-0753},
  url = {https://doi.org/10.4271/2026-99-0753}
}
```

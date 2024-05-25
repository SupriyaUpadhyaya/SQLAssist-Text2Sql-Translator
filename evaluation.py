import os
import time
import torch
import pandas as pd
import seaborn as sns
from utils import utils
import matplotlib.pyplot as plt
from datasets import load_dataset
from rouge_score import rouge_scorer
from unsloth import FastLanguageModel
from nltk.translate.bleu_score import sentence_bleu
from transformers import AutoTokenizer, LlamaForCausalLM
from utils.templates import user_message_template, text_to_sql_inference_tmpl_str, system_message

# Define constants
DATASET_IDS = [
    'determined-ai/text-to-sql-easy',
    'determined-ai/text-to-sql-medium',
    'determined-ai/text-to-sql-hard'
]
OUTPUT_DIR = 'results/'
EXCEL_FILE = 'results/eval_result.xlsx'

model, tokenizer = utils.load_model()

def load_and_prepare_data(dataset_ids):
    dfs = []
    for dataset_id in dataset_ids:
        for split in ["test", "valid"]:
            df = pd.DataFrame(load_dataset(dataset_id, split=split))
            df['Complexity'] = dataset_id.split('-')[-1]
            dfs.append(df)
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df["prompt"] = merged_df.apply(lambda x: generate_prompt(x), axis=1)
    return merged_df

def _generate_prompt_sql(inputt, context, dialect="sqlite", output="", messages=""):
    user_message = user_message_template.format(dialect=dialect, inputt=inputt, context=context, messages=messages)
    return text_to_sql_inference_tmpl_str.format(system_message=system_message, user_message=user_message)

def generate_prompt(data_point):
    return _generate_prompt_sql(
        data_point["instruction"],
        data_point["input"],
        dialect="sqlite"
    )

def evaluate_model(merged_df):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    initial_start = time.time()

    for index, row in merged_df.iterrows():
        # print("Index:", index)
        start = time.time()
        inputs = tokenizer(row['prompt'], return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
        input_length = inputs["input_ids"].shape[1]
        response = tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=True)
        query = response[0]

        merged_df.at[index, 'generated_query'] = query
        bleu = sentence_bleu([row['response'].split()], query.split())
        rouge = scorer.score(row['response'], query)
        rouge_score = (rouge['rouge1'].fmeasure + rouge['rouge2'].fmeasure + rouge['rougeL'].fmeasure) / 3
        merged_df.at[index, 'bleu_score'] = bleu
        merged_df.at[index, 'rouge_score'] = rouge_score

    merged_df.to_excel(EXCEL_FILE, index=False)
    # print("Inference time =", time.time() - initial_start)
    return merged_df

def calculate_mean_scores(merged_df):
    mean_bleu_score = merged_df['bleu_score'].mean()
    mean_rouge_score = merged_df['rouge_score'].mean()
    print("Mean BLEU Score:", mean_bleu_score)
    print("Mean ROUGE Score:", mean_rouge_score)
    return mean_bleu_score, mean_rouge_score

def calculate_scores_by_complexity(merged_df, dataset_ids):
    scores = {}
    for dataset_id in dataset_ids:
        complexity = dataset_id.split('-')[-1]
        print("Complexity:", complexity)
        mean_bleu = merged_df.loc[merged_df['Complexity'] == complexity, 'bleu_score'].mean()
        mean_rouge = merged_df.loc[merged_df['Complexity'] == complexity, 'rouge_score'].mean()
        print("Mean BLEU Score:", mean_bleu)
        print("Mean ROUGE Score:", mean_rouge)
        scores[complexity] = (mean_bleu, mean_rouge)
    return scores

def plot_scores(merged_df):
    plt.figure(figsize=(11,6))
    sns.boxplot(x='Complexity', y='rouge_score', hue='Complexity', data=merged_df, palette='husl', showmeans=True, meanprops={'marker':'o','markerfacecolor':'white','markeredgecolor':'black','markersize':'8'})
    plt.xlabel('SQL Complexity')
    plt.ylabel('ROUGE Score')
    plt.title('Box Plot of ROUGE Score for Each SQL Complexity')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "Rouge_Score.png"))

    plt.figure(figsize=(11,6))
    sns.boxplot(x='Complexity', y='bleu_score', hue='Complexity', data=merged_df, palette='husl', showmeans=True, meanprops={'marker':'o','markerfacecolor':'white','markeredgecolor':'black','markersize':'8'})
    plt.xlabel('SQL Complexity')
    plt.ylabel('BLEU Score')
    plt.title('Box Plot of BLEU Score for Each SQL Complexity')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "Bleu_Score.png"))

    selected_df = merged_df[['bleu_score', 'rouge_score']]
    sns.boxplot(data=selected_df, showmeans=True, meanprops={'marker':'o','markerfacecolor':'white','markeredgecolor':'black','markersize':'8'})
    plt.xlabel('Score Type')
    plt.ylabel('Score')
    plt.title('Box Plot of BLEU Score and ROUGE Score')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "Mean_Score.png"))

def main():
    merged_df = load_and_prepare_data(DATASET_IDS)
    evaluated_df = evaluate_model(merged_df)
    calculate_mean_scores(evaluated_df)
    calculate_scores_by_complexity(evaluated_df, DATASET_IDS)
    plot_scores(evaluated_df)

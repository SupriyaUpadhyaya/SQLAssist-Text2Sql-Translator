from langchain.memory import ChatMessageHistory
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities.sql_database import SQLDatabase
import sqlite3
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoTokenizer
import torch
import numpy as np
from datasets import load_dataset
from unsloth import FastLanguageModel

history = ChatMessageHistory()

def _generate_prompt_sql(input, context, dialect="sqlite", output="", messages=""):
    system_message = f"""You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables.

You must output the SQL query that answers the question. Use the previous conversation to answer the follow up questions. Do not provide any explanation

    """
    user_message = f"""### Dialect:
{dialect}

### Input:
{input}

### Context:
{context}

### Previous Conversation:
{messages}

"""
    if output:
        return text_to_sql_tmpl_str.format(
            system_message=system_message,
            user_message=user_message,
            response=output,
        )
    else:
        return text_to_sql_inference_tmpl_str.format(
            system_message=system_message, user_message=user_message
        )

def generate_prompt(data_point):
    full_prompt = _generate_prompt_sql(
        data_point["question"],
        data_point["context"],
        dialect="sqlite",
        # output=data_point["sql"],
    )
    return full_prompt

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# Define the dataset ID and load the dataset
dataset_id = "hoangphu7122002ai/text2sql_en"
data = load_dataset(dataset_id, split="train")

# Convert the dataset to a pandas DataFrame
df = data.to_pandas()

# Select the first 500 rows
df_subset = df.tail(500)

# Export the subset to an Excel file
excel_file = "text2sql_dataset_subset.xlsx"
df_subset.to_excel(excel_file, index=False)

print("Exported dataset subset to", excel_file)

excel_file = "text2sql_dataset_subset.xlsx"
#category_sheet_name = "window functions"  # Change this to the actual sheet name in your Excel file

# Load the category1 sheet into a DataFrame
category_df = pd.read_excel(excel_file)

# Store the result in a new "text" column
category_df["text"] = category_df.apply(lambda x: generate_prompt(x), axis=1)

tokenizer = AutoTokenizer.from_pretrained("basavaraj/text2sql-Llama3-8b")
model = LlamaForCausalLM.from_pretrained(
    "basavaraj/text2sql-Llama3-8b",
    load_in_4bit=True,
    torch_dtype=torch.float16,)
FastLanguageModel.for_inference(model)

import time

bleu_scores = []
rouge_scores = []
exact_matches = 0
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
initial_start = time.time()
Inference_time = 0

# Iterate over each row in the category_df
for index, row in category_df.iterrows():
    print("index : ", index)
    start = time.time()
    inputs = row['text']
    expected_query = row['answer']

    # Tokenize inputs
    inputs = tokenizer(inputs, return_tensors="pt").to("cuda")

    # Generate SQL queries for the batch
    outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
    input_length = inputs["input_ids"].shape[1]
    response = tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=True)
    query = response[0]
    print("Verification", response[0])

    exact_matches += sum(1 for query, expected_query in zip(query, row['answer']) if query == expected_query)

    category_df.loc[category_df['answer'] == expected_query, 'generated_query'] = query
    category_df.loc[category_df['answer'] == expected_query, 'exact_match'] = exact_matches

    bleu = sentence_bleu([expected_query.split()], query.split())
    bleu_scores.append(bleu)
    rouge = scorer.score(expected_query, query)
    rouge_score = (rouge['rouge1'].fmeasure + rouge['rouge2'].fmeasure + rouge['rougeL'].fmeasure) / 3
    rouge_scores.append(rouge_score)

    category_df.loc[category_df['answer'] == expected_query, 'bleu_score'] = bleu
    category_df.loc[category_df['answer'] == expected_query, 'rouge_score'] = rouge_score

    Inference_time += time.time() - start

    print("Inference time =", Inference_time)
#category_df['Bleu Score'] = bleu_scores
#category_df['Rouge Score'] = rouge_scores

# Save the modified DataFrame back to the Excel file
category_df.to_excel(excel_file, index=False)
# Print the total number of exact matches for the category
#print(f"Total exact matches for {category_sheet_name}:", exact_matches)
#print("Bleu score:", bleu_scores)
#print("Rogue_Score", rouge_scores)
print("Inference time =", time.time() - initial_start)

excel_file = "text2sql_dataset_subset.xlsx"
df = pd.read_excel(excel_file)
mean_BS = df['bleu_score'].mean()
mean_RS = df['rouge_score'].mean()
print(mean_BS, mean_RS)

e = "aggregation.xlsx"
df1 = pd.read_excel(e)
mbs = df1['bleu_score'].mean()
mrs = df1['rouge_score'].mean()
print(mbs,mrs)

e1 = "CTEs.xlsx"
df2 = pd.read_excel(e1)
mbs1 = df2['bleu_score'].mean()
mrs1 = df2['rouge_score'].mean()
print(mbs1,mrs1)

e2 = "multiple_joins.xlsx"
df3 = pd.read_excel(e2)
mbs2 = df3['bleu_score'].mean()
mrs2 = df3['rouge_score'].mean()
print(mbs2,mrs2)

e3 = "set operations.xlsx"
df4 = pd.read_excel(e3)
mbs3 = df4['bleu_score'].mean()
mrs3 = df4['rouge_score'].mean()
print(mbs3,mrs3)

e4 = "single join.xlsx"
df5 = pd.read_excel(e4)
mbs4 = df5['bleu_score'].mean()
mrs4 = df5['rouge_score'].mean()
print(mbs4,mrs4)

e5 = "subqueries.xlsx"
df6 = pd.read_excel(e5)
mbs5 = df6['bleu_score'].mean()
mrs5 = df6['rouge_score'].mean()
print(mbs5,mrs5)

e6 = "window functions.xlsx"
df7 = pd.read_excel(e6)
mbs6 = df7['bleu_score'].mean()
mrs6 = df7['rouge_score'].mean()
print(mbs6,mrs6)


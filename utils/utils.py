import os
import torch
import sqlite3
import streamlit as st
from unsloth import FastLanguageModel
from core.context_retriever import ContextRetriever
from utils.templates import refiner_template
from core.refiner import Refiner
from core.rephraser import Rephraser
from langchain.memory import ChatMessageHistory
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoTokenizer
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from utils.templates import user_message_template, text_to_sql_inference_tmpl_str, system_message

# Set a default model
def load_model():
    max_seq_length = 2048 # Choose any! Unsloth auto supports RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
    
    tokenizer = AutoTokenizer.from_pretrained("basavaraj/text2sql-Llama3-8b")
    model = LlamaForCausalLM.from_pretrained(
    "basavaraj/text2sql-Llama3-8b",
    load_in_4bit=True,
    torch_dtype=dtype)
    
    FastLanguageModel.for_inference(model)
    
    st.session_state["model"] = model
    st.session_state["tokenizer"] = tokenizer
    st.session_state["contextRetriever"] = ContextRetriever()
    st.session_state["refiner"] = Refiner(tokenizer=tokenizer, model=model)
    st.session_state["rephraser"] = Rephraser(tokenizer=tokenizer, model=model)

def execute_sql(db_path, sql, question) -> dict:
    conn = sqlite3.connect(db_path)
    conn.text_factory = lambda b: b.decode(errors="ignore")
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        result = cursor.fetchall()
        return {
            "question": question,
            "sql": str(sql),
            "data": result,
            "sqlite_error": "",
            "exception_class": ""
        }
    except sqlite3.Error as er:
        return {
            "question": question,
            "sql": str(sql),
            "sqlite_error": str(' '.join(er.args)),
            "exception_class": str(er.__class__)
        }
    except Exception as e:
        return {
            "question": question,
            "sql": str(sql),
            "sqlite_error": str(e.args),
            "exception_class": str(type(e).__name__)
        }
        


def _generate_prompt_sql(inputt, context, dialect="sqlite", output="", messages=""):
    user_message = user_message_template.format(dialect=dialect, inputt=inputt, context=context, messages=messages)
    return text_to_sql_inference_tmpl_str.format(system_message=system_message, user_message=user_message)

def get_selected_tables(contextRetriever, question):
    object_retriever = contextRetriever.get_object_retriever(st.session_state.topk)
    table_schema_objs = object_retriever.retrieve(question)
    table_names = [obj.table_name for obj in table_schema_objs]
    return table_names, table_schema_objs

def get_inference_prompt(follow_up, question, context, prev_hist):
    if follow_up:
        text2sql_tmpl_str = _generate_prompt_sql(
            question, context, dialect="sqlite", output="", messages=prev_hist
        )
    else:
        text2sql_tmpl_str = _generate_prompt_sql(
            question, context, dialect="sqlite", output="", messages=''
        )  
    return text2sql_tmpl_str

def generate_sql(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors = "pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
    input_length = inputs["input_ids"].shape[1]
    response = tokenizer.batch_decode(
      outputs[:, input_length:], skip_special_tokens=True
    )
    return response[0].split("\n")[0]

def update_history(exec_result):
    if 'data' in exec_result:
        if len(st.session_state.history.messages) == 2:
            st.session_state.history.messages.pop()
            st.session_state.history.messages.pop()
        st.session_state.history.add_user_message(exec_result['question'])
        st.session_state.history.add_ai_message(exec_result['sql'])

def write_log(question, selected_tables, exec_result, answer, messages, is_refined, refined_generations):
    log_string = (
        f"```\nUser Question: {question}\n"
        f"Selected Tables: {selected_tables}\n"
        f"Generated SQL Query: {exec_result.get('sql', '')}\n"  # Use get to avoid KeyError if 'sql' is missing
    )
    if 'data' in exec_result:
        log_string += f"SQL Result: {exec_result['data']}\n"
    else:
        log_string += f"SQL Error: {exec_result['sqlite_error']}\n"
    log_string += (
        f"Answer: {answer}\n"
        f"Previous conversation : {messages}\n"
        f"Is refined: {is_refined}\n"
        f"Refined queries: {refined_generations}\n```"
    )

    with open("app_logs.log", "a", buffering=1) as logfile:
        log_string_end = log_string + f"===========================================================\n```"
        logfile.write(log_string_end)

    return log_string

def get_history():
    if 'history' not in st.session_state:
        st.session_state.history = ChatMessageHistory()
          
    return st.session_state.history.messages

def transcribe(question):
    
    prev_hist = get_history()
    
    selected_table_names, selected_table_schema_objs = get_selected_tables(st.session_state.contextRetriever, question)
    
    context = st.session_state.contextRetriever.get_table_context_and_rows_str(question, selected_table_schema_objs)
    
    prompt = get_inference_prompt(st.session_state.follow_up, question, context, prev_hist)

    generated_sql = generate_sql(prompt, st.session_state.tokenizer, st.session_state.model)
    
    exec_result = execute_sql(st.session_state.db_path, generated_sql, question)
    
    is_refined, refined_generations, refined_exec_result = st.session_state.refiner.refine(generated_sql, exec_result, context, exec_result)

    rephrased_answer = st.session_state.rephraser.rephrase(refined_exec_result)

    log_content = write_log(question, selected_table_names, exec_result, rephrased_answer, prev_hist, is_refined, refined_generations)  
    
    st.session_state.current_log = log_content
    
    update_history(exec_result)

    st.session_state.query = generated_sql
    
    return rephrased_answer
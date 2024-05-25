import sqlite3
import streamlit as st
from utils.templates import list_answer_prompt, nonlist_answer_prompt

class Rephraser():
  
  def __init__(self, tokenizer, model):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
  
  def isListRequested(self, question):
        return True if "List" in question else False

  def rephrase(self,
               exec_result) -> str:
        
    if 'data' in exec_result and len(exec_result['data']) > 0 and exec_result['data'][0] != (None,):
        if self.isListRequested:
            answer_prompt = list_answer_prompt.format(exec_result)
        else:
            answer_prompt = nonlist_answer_prompt.format(exec_result)
        inputs = self.tokenizer(answer_prompt, return_tensors = "pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens = 64)
        input_length = inputs["input_ids"].shape[1]
        response = self.tokenizer.batch_decode(
            outputs[:, input_length:], skip_special_tokens=True
        )
        answer = response[0]
        print("Answer :", response)
    else:
      answer = "Sorry, could not retrive the answer. Please rephrase your question more accurately."
    return answer
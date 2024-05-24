import sqlite3
import streamlit as st
from utils.templates import refiner_template

class Refiner():
  
  def __init__(self, tokenizer, model):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
    
  
  def is_need_refine(self, exec_result: dict):
        # if self.dataset_name == 'worlddb':
        if 'data' not in exec_result:
            return True
        return False
        
        data = exec_result.get('data', None)
        if data is not None:
            if len(data) == 0:
                exec_result['sqlite_error'] = 'no data selected'
                return True
            for t in data:
                for n in t:
                     if n is None:  # fixme fixme fixme fixme fixme
                        exec_result['sqlite_error'] = 'exist None value, you can add `NOT NULL` in SQL'
                        return True
            return False
        else:
            return True

  def refine(self,
               query: str,
               evidence:str,
               schema_info: str,
               error_info: dict,
               retry = 5,
               fk_info = "") -> dict:
        
        count = 0
        is_refined = False
        refined_generations = []
        while count <= retry:
            is_refine_required = self.is_need_refine(exec_result)
            if is_refine_required:
                is_refined = True
                sql_arg = error_info.get('sql')
                sqlite_error = error_info.get('sqlite_error')
                exception_class = error_info.get('exception_class')
                prompt = refiner_template.format(query=query, evidence=evidence, desc_str=schema_info, \
                                            fk_str=fk_info, sql=sql_arg, sqlite_error=sqlite_error, \
                                                exception_class=exception_class)

                #word_info = extract_world_info(self._message)
                inputs = self.tokenizer(prompt, return_tensors = "pt").to("cuda")
                outputs = self.model.generate(**inputs, max_new_tokens = 64, use_cache = True)
                input_length = inputs["input_ids"].shape[1]
                response = self.tokenizer.batch_decode(
                outputs[:, input_length:], skip_special_tokens=True)
                query = response[0]                
                refined_generations.append(query_generated)
                exec_result = utils.execute_sql(sql=query_generated, question=question)
                count += 1
            else:
                count = retry + 1
        
        return exec_result
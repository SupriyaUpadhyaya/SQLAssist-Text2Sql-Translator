from unsloth import FastLanguageModel
from langchain.memory import ChatMessageHistory
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

db = SQLDatabase.from_uri("sqlite:////content/worlddb.db", sample_rows_in_table_info=2)
# print(db.table_info)

text_to_sql_tmpl_str = """\
### Instruction:\n{system_message}{user_message}\n\n### Response:\n{response}"""

text_to_sql_inference_tmpl_str = """\
### Instruction:\n{system_message}{user_message}\n\n### Response:\n"""

history = ChatMessageHistory()

max_seq_length = 2048 # Choose any! Unsloth auto supports RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

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

    ### Response:
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

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "supriyaupadhyaya/llama-3-8b-bnb-4bit-text-to-sql",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)
FastLanguageModel.for_inference(model)

question = "How many cities in the USA?"
context = db.table_info
messages = history.messages
text2sql_tmpl_str = _generate_prompt_sql(
        question, context, dialect="sqlite", output="", messages=messages
    )

text2sql_tmpl_str

inputs = tokenizer(text2sql_tmpl_str, return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
input_length = inputs["input_ids"].shape[1]
response = tokenizer.batch_decode(
      outputs[:, input_length:], skip_special_tokens=True
    )
query = response[0]

# Print the generated SQL query.
print(query)
history.add_user_message(question)
history.add_ai_message(query)
if len(messages) > 10:
  messages.pop()
  messages.pop()

execute_query = QuerySQLDataBaseTool(db=db)
answer = execute_query.invoke(response[0])
answer

refiner_template = """
【Instruction】
When executing SQL below, some errors occurred, please fix up SQL based on query and database info.
Solve the task step by step if you need to. Using SQL format in the code block, and indicate script type in the code block.
When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.
【Constraints】
- In `SELECT <column>`, just select needed columns in the 【Question】 without any unnecessary column or value
- In `FROM <table>` or `JOIN <table>`, do not include unnecessary table
- If use max or min func, `JOIN <table>` FIRST, THEN use `SELECT MAX(<column>)` or `SELECT MIN(<column>)`
- If [Value examples] of <column> has 'None' or None, use `JOIN <table>` or `WHERE <column> is NOT NULL` is better
- If use `ORDER BY <column> ASC|DESC`, add `GROUP BY <column>` before to select distinct values
【Query】
-- {query}
【Evidence】
{evidence}
【Database info】
{desc_str}
【Foreign keys】
{fk_str}
【old SQL】
```sql
{sql}
```
【SQLite error】
{sqlite_error}
【Exception class】
{exception_class}

Now please fixup old SQL and generate new SQL again.
【correct SQL】
"""

from core.utils import parse_json, parse_sql_from_string, add_prefix, load_json_file, extract_world_info, is_email, is_valid_date_column
import sqlite3

class Refiner():

  def __init__(self, data_path: str, dataset_name: str):
        super().__init__()
        self.data_path = data_path  # path to all databases
        self.dataset_name = dataset_name
        #self._message = {}

  def _execute_sql(self, sql: str, question: str) -> dict:
        # Get database connection
        db_path = self.data_path
        conn = sqlite3.connect(db_path)
        conn.text_factory = lambda b: b.decode(errors="ignore")
        cursor = conn.cursor()
        try:
            cursor.execute(sql)
            result = cursor.fetchall()
            return {
                "question": question,
                "sql": str(sql),
                "data": result[:5],
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

  def _is_need_refine(self, exec_result: dict):
        # spider exist dirty values, even gold sql execution result is None
        if self.dataset_name == 'worlddb':
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

  def _refine(self,
               query: str,
               evidence:str,
               schema_info: str,
               fk_info: str,
               error_info: dict) -> dict:

        sql_arg = add_prefix(error_info.get('sql'))
        sqlite_error = error_info.get('sqlite_error')
        exception_class = error_info.get('exception_class')
        prompt = refiner_template.format(query=query, evidence=evidence, desc_str=schema_info, \
                                       fk_str=fk_info, sql=sql_arg, sqlite_error=sqlite_error, \
                                        exception_class=exception_class)

        #word_info = extract_world_info(self._message)
        inputs = tokenizer(text2sql_tmpl_str, return_tensors = "pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
        input_length = inputs["input_ids"].shape[1]
        response = tokenizer.batch_decode(
        outputs[:, input_length:], skip_special_tokens=True)
        query = response[0]
        return query

count = 0
refiner = Refiner(data_path="/content/worlddb.db", dataset_name='worlddb')
query_generated = query
exec_result = refiner._execute_sql(sql=query_generated, question=question)
print(exec_result)
while count <= 5:
  is_refine_required = refiner._is_need_refine(exec_result=exec_result)
  print("is_refine_required :", is_refine_required)
  if is_refine_required:
    print("In if condiition")
    query_generated = refiner._refine(query=query_generated, evidence=exec_result, schema_info=db.table_info, fk_info="", error_info=exec_result)
    exec_result = refiner._execute_sql(sql=query_generated, question=question)
    print(exec_result)
    count += 1
    print(query_generated)
  else:
    print("in else condition")
    count = 6

answer_prompt = f'''Given the following user question, corresponding SQL query, and SQL result, answer the user question in a sentence.

 Question: {exec_result['question']}
 SQL Query: {exec_result['sql']}
 SQL Result: {exec_result['data']}
 Answer:'''

inputs = tokenizer(answer_prompt, return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
input_length = inputs["input_ids"].shape[1]
response = tokenizer.batch_decode(
      outputs[:, input_length:], skip_special_tokens=True
    )

response[0]
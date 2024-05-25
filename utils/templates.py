text_to_sql_tmpl_str = """\### Instruction:\n{system_message}{user_message}\n\n### Response:\n{response}"""


text_to_sql_inference_tmpl_str = """\### Instruction:\n{system_message}{user_message}### Response:\n"""


system_message = f"""You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables.

You must output the SQL query that answers the question. Do not provide any explanation

"""

user_message_template = """### Dialect:
{dialect}

### Input:
{inputt}

### Context:
{context}

### Previous Conversation:
{messages}

"""

list_answer_prompt = '''Given the user question, corresponding SQL query, and SQL result, answer the user question.

Here is a typical example:

Question: List name and population of the 5 cities in country with Italian language?
SQL Query: SELECT Name, Population FROM city WHERE CountryCode IN (SELECT Code FROM country WHERE Name = 'Italy') ORDER BY Population DESC LIMIT 5
SQL Result: [('Roma', 2643581), ('Milano', 1300977), ('Napoli', 1002619), ('Torino', 903705), ('Palermo', 683794)]
Answer: Here's the list of 5 cities in country with Italian language
1. Roma, 2643581
2. Milano, 1300977
3. Napoli,1002619
4. Torino, 903705
5. Palermo, 683794

Here is a new example, please start answering:

Question: {question}
SQL Query: {sql}
SQL Result: {data}
Answer:'''


nonlist_answer_prompt = '''Given the following user question, corresponding SQL query, and SQL result, answer the user question in a sentence.
 Question: {question}
 SQL Query: {sql}
 SQL Result: {data}
 Answer:'''


refiner_template = """
### Instruction:
When executing SQL below, some errors occurred, please fix up SQL based on query and database info.
Solve the task step by step if you need to. Using SQL format in the code block, and indicate script type in the code block.
When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.
### Constraints:
- In `SELECT <column>`, just select needed columns in the 【Question】 without any unnecessary column or value
- In `FROM <table>` or `JOIN <table>`, do not include unnecessary table
- If use max or min func, `JOIN <table>` FIRST, THEN use `SELECT MAX(<column>)` or `SELECT MIN(<column>)`
- If [Value examples] of <column> has 'None' or None, use `JOIN <table>` or `WHERE <column> is NOT NULL` is better
- If use `ORDER BY <column> ASC|DESC`, add `GROUP BY <column>` before to select distinct values
### Query:
-- {query}
### Evidence:
{evidence}
### Database info:
{desc_str}
### old SQL:
{sql}
### SQLite error: 
{sqlite_error}
### Exception class:
{exception_class}

Now please fixup old SQL and generate new SQL again.
### correct SQL:
"""

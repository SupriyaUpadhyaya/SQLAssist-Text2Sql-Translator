# SQLAssist Framework: Core Modules🔑

### [Context Retriever](context_retriever.py)
The Context Retriever module is designed to optimize querying and interaction with large databases by segmenting them into smaller subdatabases and filtering out unnecessary information. This module achieves efficient data handling through two main functionalities: 
1. The Query-Time Table Retrieval component stores large table schemas in an index, allowing the module to retrieve the relevant schema only when needed, thus managing large context sizes effectively. 
2. The Query-Time Sample Row Retrieval component enhances query accuracy by embedding and indexing all rows of each table and retrieving the most contextually relevant rows for any given query. 

##
### [Refiner](refiner.py)
The Refiner module is a critical component for enhancing Text-To-SQL tasks by detecting and correcting erroneous SQL queries. 

When an SQL query is received, the Refiner evaluates it for syntactical correctness and verifies that it returns non-empty results from the database. 
* If the query is valid, the module converts it into a natural language response. 
* If the query is incorrect, the Refiner iteratively refines and corrects the query based on the identified errors, re-evaluating it up to five times or until the query returns the expected result. 

##
### [Rephraser](rephraser.py)
The Rephraser module enhances the usability of NL2SQL models by converting SQL query results into clear, natural language responses. 
* It determines whether the user's question requests a list and uses appropriate prompt templates to generate the response. 
* If the query results are valid, it returns a natural language response; otherwise, it advises the user to rephrase their question for better accuracy. 

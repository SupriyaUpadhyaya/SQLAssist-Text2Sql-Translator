# SQLAssist: A RAG unified framework for Natural Language to SQL translation
Natural Language SQL Translator with following features and capabilities:

- Includes query-relevant context in instruction prompt
- Can answer follow-up questions using memory
- Refines SQL execution errors
- Rephrases answers for enhanced clarity

![Architecture of SQLAssist for text-to-sql to natural language conversion](static/hcnlp.png)


You can use our checkpoint to evaluation directly or finetune using our notebook on google colab.

1. Folder [core](core) contains code to the core modules of our framework.
2. Folder [db](db) contains the test db that the app uses and also the index files.
3. Folder [notebooks](notebooks) contains code to train, inference, evaluate and to run the application on Google Colab.
4. Folder [results](results) contains evaluation results.
5. Folder [utils](utils) contains templates and helper code.
6. Folder `static` contains files used in readme.

### Training

Google Colab notebooks are available at [notebooks](notebooks)

### Direct Evaluation

```python evaluation.py```

Evaluation results are saved in `results` 

### SQLAssist application

Start the SQLAssist application on Google Colab using [app.ipynb](SupriyaUpadhyaya/SQLAssist-Text2Sql-Translator/notebooks/app.ipynb)

To run it locally, 

```streamlit run app.py```

![inference via streamlit app](static/app.png)



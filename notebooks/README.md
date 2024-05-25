# Google Colab Notebooks

The training and inference was performed using the FastLangaugeModel module from unsloth🦥 library which accelerates the training and inference by 2 fold. 

We have fine-tuned the open-source 🦙Llama-3-8b 4 bit quantized model optimized for finetuning provided by unsloth.

##
### Training Dataset
Our selection was trained on [Clinton/Text-to-sql-v1](https://huggingface.co/datasets/Clinton/Text-to-sql-v1) dataset with 262,208 instances. It is composed of multiple datasets like wikiSQL, sql-create-context, nvbench, mimicsql-data, squall, mimic-iii, sede, eicu, spider, atis and advising. 

##
### Training⚙️
For instruction fine-tuning use [train.ipynb](SupriyaUpadhyaya/SQLAssist-Text2Sql-Translator/notebooks/train.ipynb). 

##
### Inference🔍 
For inference use [inference.ipynb](SupriyaUpadhyaya/SQLAssist-Text2Sql-Translator/notebooks/inference.ipynb). 

##
### Evaluation📈
For evaluation use [evaluation.ipynb](SupriyaUpadhyaya/SQLAssist-Text2Sql-Translator/notebooks/evaluation.ipynb). 

##
### SQLAssist Finetuned Model
Our Llama-3-8b finetuned model can be downloaded from huggingface [basavaraj/text2sql-Llama3-8b](https://huggingface.co/basavaraj/text2sql-Llama3-8b)

##
### App
To run the app on google colab, use [app.ipynb](SupriyaUpadhyaya/SQLAssist-Text2Sql-Translator/notebooks/app.ipynb). 

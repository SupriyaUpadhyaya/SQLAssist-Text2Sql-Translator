# Google Colab Notebooks

The training and inference was performed using the FastLangaugeModel module from unsloth library which accelerates the training and inference by 2 fold. We have fine-tuned open-source Llama-3-8b model using 4 bit quantized optimized for finetuning provided by unsloth. [Clinton/Text-to-sql-v1](https://huggingface.co/datasets/Clinton/Text-to-sql-v1) dataset with 262,208 instances is used. The training was done for 100 steps with learning rate of 2e-4 and adamw-8bit optimizer with a batch size of 8. The model was finetuned for q, k,v,0,gate,up and down projection layers. The total training parameters were 41,943,040. The training and validation loss for 100 steps of training were around 0.4 for both.

Our finetuned model can be downloaded from huggingface [basavaraj/text2sql-Llama3-8b](https://huggingface.co/basavaraj/text2sql-Llama3-8b)

### Training
For instruction fine-tuning use [train.ipynb](SupriyaUpadhyaya/SQLAssist-Text2Sql-Translator/notebooks/train.ipynb). 

##
### Inference 
For inference use [inference.ipynb](SupriyaUpadhyaya/SQLAssist-Text2Sql-Translator/notebooks/inference.ipynb). 

##
### Evaluation 
For evaluation use [evaluation.ipynb](SupriyaUpadhyaya/SQLAssist-Text2Sql-Translator/notebooks/evaluation.ipynb). 

##
### App 
To run the app use [app.ipynb](SupriyaUpadhyaya/SQLAssist-Text2Sql-Translator/notebooks/app.ipynb). 

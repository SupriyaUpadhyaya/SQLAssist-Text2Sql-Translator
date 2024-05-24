import torch
from trl import SFTTrainer
from unsloth import FastLanguageModel
from datasets import load_dataset, Dataset
from transformers import TrainingArguments

# Load the base model
max_seq_length = 2048 # Choose any! Unsloth auto supports RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# Load the base model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Suggested values: 8, 16, 32, 64, 128
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,  # Optimized value
    bias="none",  # Optimized value
    use_gradient_checkpointing="unsloth",  # "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # Support for rank stabilized LoRA
    loftq_config=None,  # Support for LoftQ
)

# Data Preparation
text_to_sql_tmpl_str = """\
### Instruction:\n{system_message}{user_message}\n\n### Response:\n{response}"""

text_to_sql_inference_tmpl_str = """\
### Instruction:\n{system_message}{user_message}\n\n### Response:\n"""

def _generate_prompt_sql(input, context, dialect="sqlite", output=""):
    system_message = f"""You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables.

    You must output the SQL query that answers the question.

        """
        user_message = f"""### Dialect:
    {dialect}

    ### Input:
    {input}

    ### Context:
    {context}

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
            system_message=system_message, 
            user_message=user_message
        )


def generate_prompt(data_point):
    full_prompt = _generate_prompt_sql(
        data_point["instruction"],
        data_point["input"],
        dialect="sqlite",
        output=data_point["response"],
    )
    EOS_TOKEN = tokenizer.eos_token
    return full_prompt + EOS_TOKEN

# Load and format the dataset for fine-tuning
dataset_id = "Clinton/Text-to-sql-v1"
data = load_dataset(dataset_id, split="train")
df = data.to_pandas()

# Generate prompts for training
df["text"] = df.apply(lambda x: generate_prompt(x), axis=1)
formatted_data = Dataset.from_pandas(df)

# Train the model
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_data,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # Can make training 5x faster for short sequences
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=100,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)

# Show GPU memory stats before training
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# Begin Training
trainer_stats = trainer.train()

# Show GPU memory stats after training
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# Save finetuned models
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
model.save_pretrained_merged("model_16", tokenizer, save_method="merged_16bit")

# Log in to Hugging Face
# !huggingface-cli login

# Enable HF Transfer to speed up the upload
# %env HF_HUB_ENABLE_HF_TRANSFER=1

# Push model to Hugging Face Hub
# model.push_to_hub_merged("basavaraj/text2sql-Llama3-8b", tokenizer, save_method="merged_16bit", token="<My_Token>")
# !huggingface-cli upload basavaraj/text2sql-Llama3-8b model_16
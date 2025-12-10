from datasets import load_dataset
from fastapi import FastAPI
from transformers import DataCollatorWithPadding, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

ds = load_dataset("rajpurkar/squad")

# Dataset Structure
print(ds)
print("\nTraining set:", ds["train"][0])
print("\nTesting set:", ds["validation"][0])

# Tokenization
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")  # Define model

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))


def tokenization(example):
    questions = example["question"]

    answer_list = example["answers"]
    tmp_answer_list1 = [subdict['text'] for subdict in answer_list]  # [[]...[]]
    answers = ["; ".join(sublist) for sublist in tmp_answer_list1]  # []

    return tokenizer(questions, answers, truncation=True, padding="max_length", max_length=256)


train_tokenized_ds = ds['train'].map(tokenization, batched=True)
valid_tokenized_ds = ds['validation'].map(tokenization, batched=True)

train_ds = train_tokenized_ds.add_column("labels", train_tokenized_ds["input_ids"])
valid_ds = valid_tokenized_ds.add_column("labels", valid_tokenized_ds["input_ids"])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Training
training_args = TrainingArguments("test-trainer", eval_strategy="epoch", fp16=True)

trainer = Trainer(model, training_args, train_dataset=train_ds, eval_dataset=valid_ds, data_collator=data_collator,
                  processing_class=tokenizer)

trainer.train()


# Text Generation
class TextGenerationRequest(trainer):
    start_word: str
    length: int


app = FastAPI()


@app.post("/generate_with_GPT2")
def generate_with_gpt2(request: TextGenerationRequest):
    encoded_input = tokenizer(request.start_word, return_tensors='pt')
    output = model.generate(**encoded_input, max_length=request.length, do_sample=True)

    return {"generated_text": "That is a great question! " + tokenizer.decode(output[0]) +
                              ". Let me know if you have any other questions"}

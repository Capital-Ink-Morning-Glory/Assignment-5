from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model directly
tokenizer = AutoTokenizer.from_pretrained("Capital-Ink-Morning-Glory/GPT2-SQuAD")
model = AutoModelForCausalLM.from_pretrained("Capital-Ink-Morning-Glory/GPT2-SQuAD")


# Text Generation
class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


app = FastAPI()


@app.post("/generate_with_GPT2")
def generate_with_gpt2(request: TextGenerationRequest):
    tokenized_input = tokenizer(request.start_word, return_tensors='pt')
    outputs = model.generate(
        input_ids=tokenized_input["input_ids"],
        attention_mask=tokenized_input["attention_mask"],
        temperature=0.9,  # Control randomness
        top_k=50,  # Avoid meaningless tokens
        top_p=0.95,  # nucleus sampling
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=request.length,
        min_new_tokens=50,
        do_sample=True,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        eos_token_id=tokenizer.eos_token_id
    )

    return {"generated_text": "That is a great question! " + tokenizer.decode(outputs[0], skip_special_tokens=True) +
                              ". Let me know if you have any other questions."}

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Loading pre-trained GPT-2 model and tokenizer
model_name = "gpt2" # Model size can be switched accordingly (e.g., "gpt2-medium")
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def generate_text(prompt, keyword, max_length=100, temperature=0.8, top_k=50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        num_return_sequences=1,
        early_stopping=True
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
     # Find a suitable position to insert the keyword
    position = len(generated_text) // 2
    generated_text = generated_text[:position] + " " + keyword + " " + generated_text[position:]
    return generated_text

prompt = "TCPA Call center"
generated_text = generate_text(prompt, keyword='call center', max_length=200)

# Print the generated text
print(generated_text)
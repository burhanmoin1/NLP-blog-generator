import random
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
model = GPT2LMHeadModel.from_pretrained("gpt2-large")

def generate_text(prompt, max_length=50, temperature=0.7, top_k=50):
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

    # Remove repeating sentences
    generated_text = ' '.join(generated_text.split("\n"))

    return generated_text

prompt = "Hi, my name is Alex Wilson, age 47, Male and I will tell you how to manage your money effectively:"

blog_post = generate_text(prompt, max_length=300)

print(blog_post)

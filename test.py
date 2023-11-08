import random
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
model = GPT2LMHeadModel.from_pretrained("gpt2-large")


def generate_author_voice(author_info, max_length=50, temperature=0.7, top_k=50):
    input_ids = tokenizer.encode(author_info, return_tensors="pt")
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
    generated_voice = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_voice

author_info = """
Author: Alex Wilson
Age: 47
Gender: Male
School Attended: University of Miami
Job: Real estate Broker and investor. I run a TCPA compliant real estate call center in Miami. Love my Dolphins.
Author Details: Real estate Broker and investor. I run a TCPA compliant real estate call center in Miami. Love my Dolphins.
Location: Florida
"""

def generate_introduction(author_info):
    details = [line.split(": ")[1].strip() for line in author_info.split("\n") if line.strip() and ":" in line]
    intro = f"Hi, my name is {details[0]}, age {details[1]}, {details[2]}."
    return intro

intro = generate_introduction(author_info)
print(intro)

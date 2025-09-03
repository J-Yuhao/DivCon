from load_llm import load_llm
import torch
from huggingface_hub import login
# login(token="input_your_hf_token")

model_name = "Qwen/Qwen3-8B"
tokenizer, model = load_llm(model_name)


def run_extract_prompt(sentence, task="quantity"):
    if task == "quantity":
        prompt = f"""You are an expert in extracting relevant information from text prompt. In this context, you will be given a sentence prompt describing an iamge, and you are expected to generate objects name and their numbers. Don't include any background or abstract objects. Don't add description to the object. Don't generate object without precise quantity. Return in the format of 'quantity object'. If there is only one of this object, you must return as '1 obj'.List all objects in singular form only. Do not use plural nouns. Focusing on the objects. \n
        Example:\n
        Sentence: Two cakes sits on a white plate in the center of a kitchen table, surrounded by colorful candles and streamers.\n
        Answer: 2 cake\n
        Sentence: five fire hydrants and four cell phones are lined up outside a building on a sunny day.\n
        Answer: 5 fire hydrant, 4 cell phone\n
        
        "Sentence: {sentence}"\n
        Answer:"""

    elif task == "spatial":
        messages = [
            {"role": "system", "content": """You are an expert in extracting relevant information from text prompt. In this context, you will be given a sentence prompt describing an iamge, and you are expected to generate objects position in the layout based on their spatial relationship. \n
            If the sentence describes only a left-right relationship, the objects must be positioned at approximately the same vertical level. 
            If the sentence describes only a top-bottom relationship, the objects must be aligned at approximately the same horizontal position. \n
            Return in the format of (object, position)."""},
            {"role": "user", "content": "A person and a cat are on the left of an airplane and on the right of a dog."},
            {"role": "assistant", "content": """
            Step-by-step reasoning on relation between each pair objects:\n
            1. person - left of - airplane\n
            2. cat - left of - airplane\n
            3. person - right of - dog\n
            4. cat - right of - dog\n
            5. Assign positions accordingly.\n
            Answer: (dog, left), (person, center), (cat, center), (airplane, right)"""},
            {"role": "user", "content": f"{sentence}"},
        ]
        prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
        )
    else:
        raise ValueError("task must be 'quantity' or 'spatial'")

    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.6,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if task=='quantity':
        answer = response.split("Answer:")[3].split("\n")[0]
    if task=='spatial':
        answer = response.split("Answer:")[2].split("\n")[0]
    return answer

if __name__ == "__main__":
    sentence = "There are five bananas over three dogs."
    structure_text = run_extract_prompt(sentence, task="quantity")
    print("Structure:", structure_text)

    sentence = "A chair is in the middle of a cat and a dog."
    structure_text = run_extract_prompt(sentence, task="spatial")
    print("Structure:", structure_text)
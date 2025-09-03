from load_llm import load_llm
import torch
from num_spa_infer import tokenizer, model, run_extract_prompt
from transformers import LogitsProcessor
from transformers import LogitsProcessorList
import pickle
import re
from word2number import w2n
from tqdm import tqdm
import os
import random
import argparse

def parse_object_counts(input_string):
    result = {}
    parts = input_string.split(',')

    for part in parts:
        part = part.strip()
        try:
            count_word, *obj_words = part.split()
            count = w2n.word_to_num(count_word.lower())
            obj = ' '.join(obj_words) 
            # only applicable for HRS_counting
            # if obj not in ['people', 'person']:
            #     result[obj] = count
            result[obj] = count
        except Exception:
            continue

    return result

def parse_spatial_relations(input_string):
    """
    Parses strings like: (cat, bottom-center), (car, bottom-left)
    Returns: list of (object, position) tuples
    """
    pattern = r"\(\s*([a-zA-Z0-9_ \-]+?)\s*,\s*([a-zA-Z\-]+)\s*\)"

    results = []
    for match in re.finditer(pattern, input_string):
        obj = match.group(1).strip().replace(' ', '_')
        position = match.group(2).strip().lower()
        results.append((obj, position))
    
    return results

class ForceObjectAfterNewline(LogitsProcessor):
    def __init__(self, forced_token_ids, trigger_token_id, object_counts):
        """
        forced_token_ids: dict[obj_label -> list of token_ids]
        trigger_token_id: list of trigger token ids
        object_counts: dict[obj_label -> insertion times]
        """
        self.forced_token_ids = forced_token_ids
        self.trigger_token_id = trigger_token_id
        self.object_counts = object_counts

        self.generated_counts = {obj: 0 for obj in object_counts}
        self.object_order = list(object_counts.keys())

        self.curr_obj_idx = 0
        self.curr_token_seq = []
        self.token_pos_in_seq = 0
        self.done_inserting = False
        self.first_step = True
        self.inserting_seq = False

    def __call__(self, input_ids, scores):
        if self.done_inserting:
            return scores

        while self.curr_obj_idx < len(self.object_order):
            curr_obj = self.object_order[self.curr_obj_idx]
            if self.generated_counts[curr_obj] < self.object_counts[curr_obj]:
                break
            else:
                self.curr_obj_idx += 1
                self.token_pos_in_seq = 0

        if self.curr_obj_idx >= len(self.object_order):
            self.done_inserting = True
            return scores

        curr_obj = self.object_order[self.curr_obj_idx]
        token_seq = self.forced_token_ids[curr_obj]

        should_insert = False
        if self.first_step:
            self.first_step = False
            self.inserting_seq = True
            should_insert = True
        elif self.inserting_seq:
            should_insert = True
        elif input_ids[0, -1].item() in self.trigger_token_id:
            self.inserting_seq = True
            should_insert = True

        if should_insert:
            token_id = token_seq[self.token_pos_in_seq]
            new_scores = torch.full_like(scores, -float("inf"))
            new_scores[:, token_id] = 0

            self.token_pos_in_seq += 1

            if self.token_pos_in_seq >= len(token_seq):
                self.token_pos_in_seq = 0
                self.generated_counts[curr_obj] += 1
                self.inserting_seq = False

            return new_scores

        return scores

def generate_layout(sentence, task="quantity"):
    if task == "quantity":
        structure_text = run_extract_prompt(sentence, task="quantity")
        object_info = parse_object_counts(structure_text)
    elif task == "spatial":
        structure_text = run_extract_prompt(sentence, task="spatial")
        object_info = parse_spatial_relations(structure_text)
    else:
        raise ValueError("task must be 'quantity' or 'spatial'")
    
    messages = [
        {"role": "system", "content": "You are an expert in visual planning objects' location in an image. In this context, you are expected to generate specific coordinate box locations (xyxy) for objects in a description, considering their relative sizes and positions and the numbers of objects. Size of image is 512*512. Focus on the objects."},
        {"role": "user", "content": "Provide box coordinates for an image with a cat in the middle of a car and a chair."},
        {"role": "assistant", "content": "cat(left) (230, 196, 297, 301), car(middle) (80, 270, 202, 270), chair(right) (341, 231, 447, 308),"},
        {"role": "user", "content": "The sizes of objects do not reflect the object's relative sizes. The first and third y coordicates of Car are the same which squeezes it into a line, that's not acceptable. Car should be larger than the chair, and chair should be larger than cat."},
        {"role": "assistant", "content": "Apologies for the mistake, I will avoid squeezing objects in the future and here are the corrected coordinates: cat(left) (245, 176, 345, 336), car(middle) (10, 128, 230, 384), chair(right) (353, 224, 498, 350)"},
        {"role": "user", "content": "Provide box coordinates for an image with three people skiing down a mountain."},
        {"role": "assistant", "content": "skis1 (51, 82, 399, 279), skis2 (288, 128, 472, 299), skis3 (27, 355, 418, 494),"},
        {"role": "user", "content": f"Provide box coordinates for an image with {sentence}"},
    ]
    main_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
    )
    input_ids = tokenizer([main_prompt], return_tensors='pt').input_ids.to(model.device)


    newline_token_id = [
                        tokenizer("),", add_special_tokens=False).input_ids[-1], 
                        tokenizer("499)", add_special_tokens=False).input_ids[-1], 
                        tokenizer("210),", add_special_tokens=False).input_ids[-1], 
                        tokenizer("111).", add_special_tokens=False).input_ids[-1],
                        tokenizer("102);", add_special_tokens=False).input_ids[-1],
                        tokenizer("0):", add_special_tokens=False).input_ids[-1]]
    
    if task == "quantity":
        forced_token_ids = {
            obj: tokenizer(f"{obj} (", add_special_tokens=False).input_ids
            for obj in object_info
        }
        object_counts = object_info

    elif task == "spatial":
        forced_token_ids = {
            f"{obj}({pos})": tokenizer(f" {obj}({pos}) ( ", add_special_tokens=False).input_ids
            for obj, pos in object_info
        }
        object_counts = {f"{obj}({pos})": 1 for obj, pos in object_info}


    logits_processor_obj = ForceObjectAfterNewline(forced_token_ids, newline_token_id, object_counts)


    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=250,
        logits_processor=LogitsProcessorList([logits_processor_obj]),
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0,
        # repetition_penalty=1.3,
        pad_token_id=tokenizer.eos_token_id
    )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text.split("assistant")[-1].strip()

def parse_layout_output(output_str):
    pattern = r"([a-zA-Z_()0-9 \-\[\]]+?)(?:\s*:\s*)?\(\s*(-?\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)\s*\)"

    labels = []
    boxes = []

    for match in re.finditer(pattern, output_str):
        label = match.group(1).strip()

        label = re.sub(r"\(.*?\)", "", label).strip()

        coords = tuple(map(int, match.groups()[1:]))
        labels.append(label)
        boxes.append(coords)

    return [labels, boxes]


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["HRS", "NSR1K"], required=True, help="Dataset name")
    parser.add_argument("--task", type=str, choices=["numerical", "spatial"], required=True, help="Type of reasoning task")
    args = parser.parse_args()

    data_file = f"{args.dataset}_{'counting' if args.task == 'numerical' else 'spatial'}.p"
    input_path = data_file
    output_path = f"./LLM_gen_layout/{data_file}"


    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    if os.path.exists('processed.txt'):
        with open('processed.txt', 'r') as f:
            processed_prompts = set(line.strip() for line in f)
    else:
        processed_prompts = set()

    for prompt in tqdm(data.keys()):
        if prompt in processed_prompts:
            continue

        try:
            layout_str = generate_layout(prompt, task='quantity')
            layout = parse_layout_output(layout_str)
            data[prompt] = layout

            with open(output_path, 'wb') as f:
                pickle.dump(data, f)

            with open('processed.txt', 'a') as f:
                f.write(prompt + '\n')

        except Exception as e:
            print(f"Error on {prompt}: {e}")
            continue
    
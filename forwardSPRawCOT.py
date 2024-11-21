import pandas as pd
import torch, accelerate
import pandas as pd
import json
#from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from tqdm import tqdm
import re
from sympy import symbols, Eq, simplify
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application,
    function_exponentiation, convert_xor
)

# Define transformations with more options
transformations = (
    standard_transformations +
    (implicit_multiplication_application, function_exponentiation, convert_xor)
)
cache_dir = "/projects/klybarge/HPV_Information_Extraction/hf_models/"

start_time = datetime.now()

model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"

model_name = model_path.split("/")[-1]

print(f"Model Name: {model_name}")

x, w, y, z = symbols('x w y z')
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    cache_dir=cache_dir,
    token="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" # this is Mine
    )
tokenizer.pad_token = tokenizer.eos_token

accelerator = Accelerator()
device = accelerator.device

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    cache_dir=cache_dir,
    device_map="auto",
    offload_folder="offload",
    torch_dtype=torch.bfloat16,
    token="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"
    )


generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "temperature":0.1,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 220
}
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            data.append({
                'sentence': item['sentence'],
                'option1': item['option1'],
                'option2': item['option2'],
                'answer': item['answer']
            })
    return data

# Example usage
df = pd.read_csv('data/SVAMP_wxyz.csv')
    
correct = 0
total = 0
predictions = []
ground_truths = []
correct_list = []
"""
Each pack of dvd costs w dollars.\n
having a discound makes the price w-x dollars.\n
Therefore price of each dvd pack is w-x dollars.\n
"""

#eg= "Sentence: Sammy wanted to go to where the people were. Where might he go?\n Option1: Race Track \n Option2: Populated Areas \n Option3: The desert \n Option4: Apartment \n Option5: Roadblock \n Process: \n 'The desert' is typically unpopulated; eliminate C. \n 'Roadblock' is a location but is not generally associated with gatherings of people;  eliminate E. \n 'Apartment' is a specific residence but doesn’t represent a general area where people gather; eliminate D. \n 'Race track' can have people, but it is specific and doesn’t fully capture a broad gathering place; eliminate A. \n 'Populated areas' directly suggests places where people gather, making it the best answer. \n Final Answer: Populated Areas"
accuracy_file = open('accuracy_progress_forward_8b.txt', 'w')

eg= f"""Example:
Q: John had 5 apples. He gave 2 to his sister. Then his mom gave him 3 more. How many apples does he have now?
A: Let's think step by step:
1. Initially had 5 apples
2. Gave away 2, so 5 - 2 = 3 apples
3. Got 3 more, so 3 + 3 = 6 apples
The answer is 6

Q: John had w apples. He gave x to his sister. Then his mom gave him y more. How many apples does he have now?
A: Let's think step by step:
1. Initially had w apples
2. Gave away x, so have w - x apples
3. Got y more, so add y
The symbolic expression is: w - x + y
"""
counts = 0
def advanced_preprocess(expr_str):
    # Remove unnecessary whitespace
    expr_str = expr_str.replace(' ', '')
    # Fix common issues with model-generated expressions
    expr_str = expr_str.replace('^', '**')  # Replace caret with exponentiation
    expr_str = expr_str.replace('×', '*')   # Replace multiplication symbol
    expr_str = expr_str.replace('÷', '/')   # Replace division symbol
    # Handle implicit multiplication
    expr_str = re.sub(r'(\d)([a-zA-Z(])', r'\1*\2', expr_str)
    expr_str = re.sub(r'([a-zA-Z)])(\d)', r'\1*\2', expr_str)
    expr_str = re.sub(r'([a-zA-Z)])([a-zA-Z(])', r'\1*\2', expr_str)
    return expr_str



def replace_symbols_with_blanks(text):
    """
    Replace each occurrence of w,x,y,z with sequential blanks
    Handles punctuation and ensures all occurrences are replaced
    
    Args:
        text (str): Input word problem text
    Returns:
        tuple: (transformed text, dict mapping blanks to original symbols)
    """
    import re
    
    # Create a regex pattern that matches w,x,y,z with optional punctuation
    # This will match isolated symbols even with punctuation
    pattern = r'\b[wxyz]\b|(?<=\s)[wxyz](?=[,.])|(?<=[,.\s])[wxyz](?=[\s,.])'
    
    blank_to_symbol = {}
    blank_count = 1
    
    def replace_match(match):
        nonlocal blank_count
        symbol = match.group(0)
        blank = f'#blank{blank_count}#'
        blank_to_symbol[blank] = symbol
        blank_count += 1
        return blank
    
    transformed = re.sub(pattern, replace_match, text)
    return transformed, blank_to_symbol
results = []
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing questions"):
    question = row['Question']
    question_blanked = replace_symbols_with_blanks(question)
    ground_truth = row['Equation'].replace(" ","")
    arithmatic_quest = row['Combined']
    arithmetic_answer = row['Answer']

    numeric_prompt=f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n You are a helpful assistant. 
    <|eot_id|><|start_header_id|>user<|end_header_id|> Please Provide just the numeric answer for the question below, nothing else: \n 
    {arithmatic_quest}\n
    Answer:
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    
    
    
   
    
    numeric_query_encoding = tokenizer.encode(numeric_prompt)
            
    # Generate the response from the model
    response_tensor_numeric = model.generate(
        torch.tensor(numeric_query_encoding).unsqueeze(dim=0).to(device),  
        **generation_kwargs
    ).squeeze()[len(numeric_query_encoding):]
    
    # Decode the response
    response_numeric = tokenizer.decode(response_tensor_numeric, skip_special_tokens=True)
    print(response_numeric)


    self_alignment_prompt = numeric_prompt=f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n You are a helpful assistant. 
    <|eot_id|><|start_header_id|>user<|end_header_id|> You are a math expert. Follow these exact steps:
    1. Think through the numeric version step by step
    2. Show your symbolic solution step by step
    3. End with "The symbolic expression is: " followed by ONLY the expression\n

    {eg} \n
    Q: {arithmatic_quest}\n
    A: {response_numeric}\n
    Q: {question}\n
    A:
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    #print(self_alignment_prompt)
    
    sa_query_encoding = tokenizer.encode(self_alignment_prompt)
    response_tensor_sa = model.generate(
        torch.tensor(sa_query_encoding).unsqueeze(dim=0).to(device),  
        **generation_kwargs
    ).squeeze()[len(sa_query_encoding):]
    
    # Decode the response
    response_sa = tokenizer.decode(response_tensor_sa, skip_special_tokens=True)
    print(response_sa)
    
    
    r = response_sa.split('\n')[-1].split(":")[-1]
    #print(r)
    #print('=================================================')
    final_response = r.lower().strip().replace(" ","")
    final_response = advanced_preprocess(final_response)
    
    ground_truth   =  ground_truth.lower().strip()
    print(f"Ground Truth: {ground_truth}, Response:{final_response}")
    print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
    results.append({
        'Question': question,
        'Ground Truth': ground_truth,
        'Final Response': final_response
    })

results_df = pd.DataFrame(results)
results_df.to_csv('outputSPRAWCOT.csv', index=False)

print("Results saved to outputSPRAWCOT.csv")
    


import openai
from utils import dispatch_openai_api_requests, return_args, save_json, read_json
from data_prepare import data_prepare
from prompts import template_icl_sum, role_str_sum
from tqdm import tqdm

config = return_args()

final_data_list, selected_keys, target_dict = data_prepare()

summary_save_dict = {}

count = 0
total = 0
prompt_list = []
for d_i, data_elem in enumerate(final_data_list[:]): 
    feature_dict = {} 
    for k in selected_keys:
        if k == 'trace_type':
            continue
        feature_dict[k] = data_elem[k]
    target = data_elem['trace_type']
    fid = data_elem['fid']
    target_idx = target_dict[target.lower()]

    messages=[
        {"role": "system", "content": role_str_sum},
        ]

    data_f_prompt =  template_icl_sum.format(feature_dict)
    messages.append({"role": "user", "content": data_f_prompt})
    prompt_list.append(messages)
print ("Data have been procssed.")

openai_responses = dispatch_openai_api_requests(prompt_list, len(prompt_list), api_batch=int(config['api_batch']), api_model_name = "gpt-3.5-turbo")

summary_save_dict = {}
for d_i, data_elem in tqdm(enumerate(final_data_list[:])): 
    target = data_elem['trace_type']
    fid = data_elem['fid']
    target_idx = target_dict[target.lower()]
    openai_response = openai_responses[d_i]
    response = openai_response['choices'][0]['message']['content']
    if fid not in summary_save_dict:
        summary_save_dict[fid] = response

save_json(summary_save_dict, "output/final_summary_save_dict_"+str(config['seed'])+".json")



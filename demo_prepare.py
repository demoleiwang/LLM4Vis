import openai
from utils import dispatch_openai_api_requests, return_args, save_json, read_json
from data_prepare import data_prepare, data_prepare_split
from prompts import role_str_demo_prepare, demos_descri, demos_descri_hint
from tqdm import tqdm
import json
import random

config = return_args()

target_dict = {'line':0, 'scatter':1, 'bar':2, 'box':3}
threshold = 0.05
summary_save_dict = read_json("output/final_summary_save_dict_"+str(config["seed"])+".json")
demo_data_list, test_data_list, selected_keys = data_prepare_split(config['seed'])
demo_data_list = demo_data_list[:]
dpi_dict = {}

demo_pool = []

for iter_ in range (6):
	prompt_list = []
	new_data_list = []
	count = 0
	total = 0
	print (f'\n==Stage 1, iter:{iter_}==')
	for d_i, data_elem in enumerate(demo_data_list[:]): 
	    total += 1

	    feature_dict = {} 
	    for k in selected_keys:
	        if k == 'trace_type':
	            continue
	        feature_dict[k] = data_elem[k]
	    target = data_elem['trace_type']
	    if target.lower() == 'line': #{'line chart': 0.5, 'scatter plot': 0, 'bar chart': 0, 'box plot': 1}
	        target_str = 'line chart'
	    elif target.lower() == 'scatter':
	        target_str = 'scatter plot'
	    elif target.lower() == 'bar':
	        target_str = 'bar chart'
	    elif target.lower() == 'box':
	        target_str = 'box plot'
	        
	    fid = data_elem['fid']
	    target_idx = target_dict[target.lower()]

	    elem_fe_de = summary_save_dict[fid]

	    if fid not in dpi_dict:
	    	dpi_dict[fid] = {'final_step':0, 'stop':0, 'iters':{}}

	    if dpi_dict[fid]['stop'] == 1:
	    	dip_elem_x = dpi_dict[fid]
	    	demos_descri.format(elem_fe_de) + "\n" +dip_elem_x['iters'][dip_elem_x['final_step']]['predctions'] + "\n\n\n"
	    	demo_pool.append(demos_descri)
	    	continue

	    dpi_dict[fid]['iters'][dpi_dict[fid]['final_step']] = {}

	    messages=[
	        {"role": "system", "content": role_str_demo_prepare},
	        ]
	    if dpi_dict[fid]['final_step'] == 0:
	    	data_f_prompt = demos_descri.format( elem_fe_de)
	    	messages.append({"role": "user", "content": data_f_prompt})
	    else:
	    	
	    	# pred_answer_x = dpi_dict[fid]['final_step']
	    	pred_answer = dpi_dict[fid]['iters'][dpi_dict[fid]['final_step']-1]['pred_answer']
	    	predctions_json = dpi_dict[fid]['iters'][dpi_dict[fid]['final_step']-1]['predctions_json']
	    	data_f_prompt = demos_descri_hint.format(target_str, \
	    		pred_answer[1][0].lower(), \
	    		predctions_json, elem_fe_de)
	    	if iter_ > 3:
	    		random.seed(config['seed'])
	    		random.shuffle(demo_pool)
	    		demo_str_x = demo_pool[0]
	    		data_f_prompt = demo_str_x+ data_f_prompt
	    	messages.append({"role": "user", "content": data_f_prompt})

	    dpi_dict[fid]['iters'][dpi_dict[fid]['final_step']]['demos_descri'] = demos_descri
	    dpi_dict[fid]['iters'][dpi_dict[fid]['final_step']]['elem_fe_de'] = elem_fe_de

	    prompt_list.append(messages)
	    new_data_list.append(data_elem)

	if len(prompt_list) == 0:
		break
	print (f"number of prompts: {len(prompt_list)}")
	openai_responses = dispatch_openai_api_requests(prompt_list, len(prompt_list), api_batch=int(config['api_batch']), api_model_name = "gpt-3.5-turbo")

	print (f'\n==Stage 2, iter:{iter_}==')
	for d_i, data_elem in tqdm(enumerate(new_data_list[:])): 
		target = data_elem['trace_type']
		if target.lower() == 'line': 
			target_str = 'line chart'
		elif target.lower() == 'scatter':
			target_str = 'scatter plot'
		elif target.lower() == 'bar':
			target_str = 'bar chart'
		elif target.lower() == 'box':
			target_str = 'box plot'

		fid = data_elem['fid']
		target_idx = target_dict[target.lower()]
		openai_response = openai_responses[d_i]
		response = openai_response['choices'][0]['message']['content']
		demos_str_x = []
		for xxx in response.split('.'):
			if 'hint' in xxx.lower() or 'previous' in xxx.lower():
				pass
			else:
				demos_str_x.append(xxx)
		predctions = ".".join(demos_str_x)

		start_index = predctions.find('{')
		end_index = predctions.find('}') + 1

		# Extract the JSON substring
		json_str = predctions[start_index:end_index]

		# Parse the JSON string into a dictionary

		try:
		    predctions_json = json.loads(json_str)
		    predctions_json['line chart']
		    predctions_json['scatter plot']
		    predctions_json['bar chart']
		    predctions_json['box plot']
		except:
		    predctions_json = {'line chart':0.25, 'scatter plot':0.25, 'bar chart':0.25, 'box plot':0.25}

		print ('predctions_json:',predctions_json, target_str)
		pred_answer = list(sorted(predctions_json.items(), key=lambda x: float(x[-1])))[-2:]
		
		dpi_dict[fid]['iters'][dpi_dict[fid]['final_step']]['pred_answer'] = pred_answer
		dpi_dict[fid]['iters'][dpi_dict[fid]['final_step']]['predctions_json'] = predctions_json
		dpi_dict[fid]['iters'][dpi_dict[fid]['final_step']]['predctions'] = predctions

		if target_str.lower() == pred_answer[1][0].lower() and pred_answer[1][1] - pred_answer[0][1]>threshold:
			dpi_dict[fid]['stop'] = 1
			# print ('stop')
		else:
			dpi_dict[fid]['final_step'] +=1
			# print ('nonstop')

dip_save_dict = {}
for fid, dip_elem in dpi_dict.items():
	if dip_elem['stop'] == 1:
		xx = dip_elem['iters'][dip_elem['final_step']]
		dip_save_dict[fid] = [xx['demos_descri'], xx['elem_fe_de'], xx['predctions']]

save_json(dpi_dict, "output/final_dip_hint_"+str(config['seed'])+"_all.json")
save_json(dip_save_dict, "output/final_dip_hint_"+str(config['seed'])+"_use.json")
print ("save done!")
# [demos_descri, elem_fe_de, predctions]
	    # print ('---',d_i)
	    # print (response)
	    # print ()




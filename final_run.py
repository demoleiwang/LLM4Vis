import openai
from utils import dispatch_openai_api_requests, return_args, save_json, read_json
from data_prepare import data_prepare, data_prepare_split
from prompts import role_str_demo_prepare, demos_descri, demos_descri_hint, demos_descri_x
from tqdm import tqdm
import json

config = return_args()

target_dict = {'line':0, 'scatter':1, 'bar':2, 'box':3}
target_dict_right = {'line':[], 'scatter':[], 'bar':[], 'box':[]}

# summary_save_dict = read_json("output/final_summary_save_dict_"+str(config["seed"])+".json")
# demo_prepare_dict = read_json("output/final_dip_hint_"+str(config["seed"])+"_use.json")
# sim_dict = read_json("output/final_sim_dict"+str(config['seed'])+".json")
summary_save_dict = read_json("output/final_summary_save_dict_2023.json")
demo_prepare_dict = read_json("output/final_demo_prepare_dict_hint_2023_z.json")
sim_dict = read_json("output/final_sim_dict2023.json")

demo_data_list, test_data_list, selected_keys = data_prepare_split(config['seed'])

return_re = {'line':0, 'scatter':0, 'bar':0, 'box':0}
return_wrong = {'line':0, 'scatter':0, 'bar':0, 'box':0}
final_results = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
d_is=[]

prompt_list = []
count = 0
total = 0

test_data_list = test_data_list[:]

for d_i, data_elem in enumerate(test_data_list[:]):
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
    near_ids = sim_dict[fid]


    demos_str = ''
    break_n = 5
    countx = 0
    for demo_elem_id in near_ids[:]:
        try:
            demo_elem = demo_prepare_dict[demo_elem_id]
            demos_str += demo_elem[0].format(demo_elem[1]) + "\n" +demo_elem[2] + "\n\n\n"
            countx +=1
            if countx >= break_n:
                break
        except:
            pass

    messages=[
        {"role": "system", "content": role_str_demo_prepare},
        ]

    data_f_prompt =  demos_str+ demos_descri_x.format( elem_fe_de)
    messages.append({"role": "user", "content": data_f_prompt})
  
    prompt_list.append(messages)

print (f"number of prompts: {len(prompt_list)}")
openai_responses = dispatch_openai_api_requests(prompt_list, len(prompt_list), api_batch=int(config['api_batch']), api_model_name = "gpt-3.5-turbo-16k")


for d_i, data_elem in tqdm(enumerate(test_data_list[:])): 
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
	
	predctions = response#".".join(demos_str_x)

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
	try:
		pred_answer = list(sorted(predctions_json.items(), key=lambda x: float(x[-1])))[-2:]
	except:
		predctions_json = {'line chart':0.25, 'scatter plot':0.25, 'bar chart':0.25, 'box plot':0.25}
		pred_answer = list(sorted(predctions_json.items(), key=lambda x: float(x[-1])))[-2:]
	
	pred_answer = "; ".join([elemxx[0] for elemxx in pred_answer])
	print ("pred_answer:", pred_answer)

	ans_f = ''
	true_false = 0
	if target.strip().lower() in pred_answer.strip().lower():
	    ans_f = 'correct'
	    true_false = 1
	    count += 1
	else:
	    ans_f = 'incorrect'
	    true_false = 0


	    

	if true_false == 1:
	    target_dict_right[target.strip().lower()].append(fid)

	    return_re[target] +=1
	    d_is.append(d_i)
	#         if shem_flag == 'FFF':
	#         demos += data_f_prompt + predctions +"\n\n\n"
	    if fid not in demo_prepare_dict:
	        demo_prepare_dict[fid] = [demos_descri, elem_fe_de, predctions]

	if "line" in  pred_answer.lower():
	    final_results[target_idx][0] += 1

	    if true_false==0:
	        return_wrong[target]+=1

	if "scatter" in  pred_answer.lower():
	    final_results[target_idx][1] += 1

	    if true_false==0:
	        return_wrong[target]+=1

	if "bar" in  pred_answer.lower():
	    final_results[target_idx][2] += 1

	    if true_false==0:
	        return_wrong[target]+=1

	if "box" in  pred_answer.lower():
	    final_results[target_idx][3] += 1

	    if true_false==0:
	        return_wrong[target]+=1

	print (f"1 confusion:{final_results}")
	print (f'Hit Ratio: {count}/{total} = {count*1.0/total}, true_false: {true_false}\n')





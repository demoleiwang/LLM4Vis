import asyncio
import openai
from tqdm import tqdm
import time
import json


openai.api_key = ""
openai.api_base = "" 


def read_json(file_name):
    with open(file_name, 'r') as f:
        return json.load(f)

def save_json(data, file_name):
    with open(file_name, "w") as outfile:
        json.dump(data, outfile, indent=2)

def return_args():
	config= {
	"seed": 2024,
	"api_batch": 20,
	}

	return config


async def dispatch_openai_requests(
        messages_list,
        model: str,
        temperature: float,
):
    async_responses = [
        openai.ChatCompletion.acreate(
            model=model,
            messages=x,
            temperature=0
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)



def dispatch_openai_api_requests(prompt_list, batch_size, api_batch, api_model_name = "gpt-3.5-turbo"):
    openai_responses = []

    for i in tqdm(range(0, batch_size, api_batch)):
        while True:
            try:
                openai_responses += asyncio.run(
                    dispatch_openai_requests(prompt_list[i:i + api_batch], api_model_name, 0)
                )
                break
            except KeyboardInterrupt:
                print(f'KeyboardInterrupt Error, retry batch {i // api_batch} at {time.ctime()}',
                      flush=True)
                time.sleep(5)
            except Exception as e:
                print(f'Error {e}, retry batch {i // api_batch} at {time.ctime()}', flush=True)
                time.sleep(5)
    return openai_responses


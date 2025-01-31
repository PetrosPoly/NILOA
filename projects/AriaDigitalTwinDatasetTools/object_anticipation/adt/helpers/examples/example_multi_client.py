from openai_multi_client import OpenAIMultiClient

# Remember to set the OPENAI_API_KEY environment variable to your API key
api = OpenAIMultiClient(endpoint="chats", data_template={"model": "gpt-4o-mini"})

params = dict(       
    model="gpt-4-turbo",
    temperature=0.35,
    top_p=0.25
)

api = OpenAIMultiClient(async_client, endpoint="chat.completions", data_template=params, max_retries=0)

def make_requests():
    for num in range(1, 10):
        api.request(data={
            "messages": [{
                "role": "user",
                "content": f"Can you tell me what is {num} * {num}?"
            }]
        }, metadata={'num': num})

api.run_request_function(make_requests)

for result in api:
    num = result.metadata['num']
    response = result.response['choices'][0]['message']['content']
    print(f"{num} * {num}:", response)
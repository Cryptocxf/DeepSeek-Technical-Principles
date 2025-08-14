from openai import OpenAI

# Configure OpenAI API to connect to vLLM  
openai_api_key = "EMPTY"  # Placeholder  
openai_api_base = "http://127.0.0.1:5000/v1"  


#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve /data/model/deepseek-r1-32b --port 5000 --max-model-len 2048 --gpu-memory-utilization 0.8 --tensor-parallel-size 8

client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)  

prompt = "My name is Zhang San. I want to write a PhD application email to Professor Li Si at Beihang University (about large model privacy research). Please help me draft a template.\nAnswer:"  

response = client.completions.create(  
    model="/data/model/deepseek-r1-32b",  
    prompt=prompt,  
    max_tokens=1000,  
    temperature=0.7,  
    frequency_penalty=1.0,  
    stream=True,  
    stop=["<|endoftext|>"]  
)  

# Stream the output  
full_response = ""  
for chunk in response:  
    if chunk.choices[0].text:  
        print(chunk.choices[0].text, end="", flush=True)  
        full_response += chunk.choices[0].text  
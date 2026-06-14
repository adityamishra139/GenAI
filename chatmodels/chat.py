from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()

model=init_chat_model("groq:openai/gpt-oss-120b",temprature=0.9,max_tokens=20) 
# this can be gpt-4o-mini instead of gpt-oss-120b. 
response=model.invoke("What is difference between langchain and langraph?")
print(response.content)

# model=init_chat_model("mistral-small-2506")
# response=model.invoke("What is difference between langchain and langraph?")
# print(response.content)

from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
load_dotenv()
model=ChatMistralAI(
    model="mistral-small-2506",
    temperature=0.9,
)
messages=[
    SystemMessage(content="You are a Funny assistant.")
    
]
print("---------enter exit or quit to stop the conversation-------")
while True:
    prompt=input("you : ")
    messages.append(HumanMessage(content=prompt))
    if prompt.lower() in ["exit","quit"]:
        break
    response=model.invoke(messages)
    messages.append(AIMessage(content=response.content))
    print("mistral : ",response.content)


print("Full message context:")
print(messages)


    
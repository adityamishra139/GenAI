from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from langchain_core.output_parsers import PydanticOutputParser
from typing import List, Optional,Counter
from dotenv import load_dotenv
load_dotenv()

model=ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.9,
)

class Complaint(BaseModel):
    issue: str
    emotion:str
    severity: str

parser=PydanticOutputParser(pydantic_object=Complaint)

prompt=ChatPromptTemplate.from_messages([
    ('system',
     """
     You are a sentiment analyser and complaint classifier assistant. You will be given a customer complaint and your task is to extract the issue, emotion and severity of the complaint.
     Classify the complaint into ONE of these issue categories ONLY:

- Delivery Issue
- Payment Issue
- App Issue
- Support Issue
- Product Issue
- Other
{format_instructions}
     
     """),
    ('human',"""
     these are the inputs from the customers:
     {complaint}
     """)
])
    
data=[
       "My order has not been delivered even after 7 days and there is no update from your side.",
    
    "I was charged twice for the same product and customer support is not helping me.",
    
    "The app keeps crashing whenever I try to make a payment.",
    
    "Delivery was late but the product quality is good.",
    
    "I received a damaged product and no one is responding to my return request.",
    
    "The website is very slow and takes too long to load pages.",
    
    "I can't log into my account even after resetting my password multiple times.",
    
    "Customer support was rude and unhelpful during my query.",
    
    "My refund is still not processed even after 10 days.",
    
    "The product description was misleading and not what I received.",
    
    "Everything works fine but sometimes the app lags a little.",
    
    "Payment failed but the amount got deducted from my account.",
    
    "The delivery executive marked the order as delivered but I never received it.",
    
    "Your app UI is confusing and hard to navigate.",
    
    "I had a great experience overall, but delivery could be faster.",
    
    "The coupon code did not work even though it was valid.",
    
    "I am extremely frustrated with the delays and lack of communication.",
    
    "The product stopped working after just 2 days of use.",
    
    "Support team resolved my issue but it took too long.",
    
    "I keep getting error messages while placing an order.",
    
    "This is the worst service I have ever experienced.",
    
    "The packaging was poor and the item arrived broken.",
    
    "The app logs me out automatically again and again.",
    
    "I tried contacting support multiple times but got no response.",
    
    "The checkout process is too complicated and confusing."
    ]
    
memory=[]
    
for complaint in data:
        final_prompt=prompt.invoke({
            "complaint":complaint,
            "format_instructions":parser.get_format_instructions()
        })
        response=model.invoke(final_prompt)
        try:
            parsed=parser.parse(response.content)
            memory.append(parsed)
        except Exception as e:
            print(f"Error parsing response for complaint: {complaint}")

# most common complaints    
complaints=[]
for mem in memory:
    complaints.append(mem.issue)
count=Counter(complaints)
sorted_count=count.most_common()
print("\n--- Most Common Complaints ---")
print(sorted_count)    

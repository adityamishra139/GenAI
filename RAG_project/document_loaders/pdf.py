from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
load_dotenv()

model=ChatMistralAI(
    model="mistral-small-2506",
    temperature=0.9
)




data=PyPDFLoader("quantum_mechanics_notes.pdf")
docs=data.load()

prompt=ChatPromptTemplate.from_messages([
    ('system',
     """
     You are a summany analyst. You need to summarise a given input pdf by the user in explainable way.
     """),
    
    
    ('human',
     """
     here is the input pdf to summarise: {data}
     """)
    
    
])


final_prompt=prompt.invoke({
    "data" : docs
})

response=model.invoke(final_prompt)
print(response.content)



### we use the recursivetext splitter to split the text into smaller chunks and then we can use the chunks to train our model.
from future import annotations
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

template=ChatPromptTemplate.from_messages([
    
    ("system","you are a AI that will summarise the text and answer the question based on the text. "    ),
    ("human","{data}")
    
])
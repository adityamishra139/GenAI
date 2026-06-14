### we use the recursivetext splitter to split the text into smaller chunks and then we can use the chunks to train our model.
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

# loading the pdf 

data =PyPDFLoader("./quantum_mechanics_notes.pdf")
docs=data.load()

# spliting the document
splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100
    )

#chunking the document into smaller chunks, and feeding it to AI models
chunks = splitter.split_documents(docs)














from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document

embedding_model = OpenAIEmbeddings()

docs=[]

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="chroma_db",
        
)

vectorstore.similarity_search("query",k=2)

# load the pdf 
# split to chunks
# create embeddings for each chunk
# store in chromadb


# from langchain_community.vectorstores import Chroma
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchain_mistralai import ChatMistralAI
# from dotenv import load_dotenv

# load_dotenv()

# # lead the pdf:
# data = PyPDFLoader("./quantum_mechanics_notes.pdf")
# docs=data.load()

# splitter=RecursiveCharacterTextSplitter(
#     chunk_size=1000, chunk_overlap=100
# )

# # split to chunks
# chunks=splitter.split_documents(docs)

# # create the embeddings for each chunk and store in chromadb
# embedding_model=OpenAIEmbeddings()

# # creating a db 
# vectorstore=Chroma.from_documents(
#     documents=chunks,
#     embedding=embedding_model,
#     persist_directory="chroma_db",
# )



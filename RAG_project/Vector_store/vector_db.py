from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()
data=PyPDFLoader("document_loaders/quantum_mechanics_notes.pdf")
docs=data.load()

splitter=RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks=splitter.split_documents(docs)

embedding_model = OpenAIEmbeddings()
vectorstore=Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="Chroma-DB"
)


result=vectorstore.similarity_search("What is the Schrödinger equation?",k=2)
for r in result: 
    print(r.page_content)
    
retriever=vectorstore.as_retriever()
docs=retriever.invoke("What is the Schrödinger equation?")
for d in docs:
    print(d.page_content)
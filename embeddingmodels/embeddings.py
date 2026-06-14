from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
embeddings=OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=64 
    # ideal case is 512 dimensional vector
)

texts=["I am data analyst","I am a data scientist","I am a software engineer"]

vector=embeddings.embed_documents(texts)
print(vector)
from langchain_community.document_loaders import WebBaseLoader
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
load_dotenv()

model = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0.9
)

url=WebBaseLoader("https://www.apple.com/in/macbook-pro/")
docs=url.load()

print(docs)

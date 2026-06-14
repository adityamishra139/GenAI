from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import PydanticOutputParser
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

class Summary(BaseModel):
    summary: str
    formulas: list[str]
    Names: list[str]
    

parser=PydanticOutputParser(pydantic_object=Summary)

data = PyPDFLoader("document_loaders/quantum_mechanics_notes.pdf")
docs = data.load()

splitter=RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks=splitter.split_documents(docs)
prompt=ChatPromptTemplate.from_messages([
    ("system",
     """
     You are an AI that summarizes the text.
     Return output in this format:
    {format_instructions}
    Extract:
- summary (short paragraph)
- formulas (only if present)
- names of scientists or concepts

Return empty list if not found.
     """),
    ("human",
     """
     here is the text to summarise: {data}
     """)
])

final_prompt=prompt.invoke({
    
    "data" : docs,
    "format_instructions" : parser.get_format_instructions()
}
    
)





model = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0.9
)

result=model.invoke(final_prompt)
parsed_result=parser.parse(result.content)

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatMessagePromptTemplate, ChatPromptTemplate, ChatMessagePromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser


load_dotenv()

model=ChatMistralAI(
    model="mistral-small-2506",
    temperature=0.9,
)


class Movie(BaseModel):
    title: str
    release_year : Optional[int]
    genre: List[str]
    director: Optional[str]
    main_cast: List[str]
    rating: Optional[float]
    summary: str

parser=PydanticOutputParser(pydantic_object=Movie)


prompt=ChatPromptTemplate.from_messages([
    ("system",
     
     """
     extract movie information in the paragraph
     {format_instructions}
     """),
    ("human","""
     {paragraph}
     """)
])

para=input("Give your Paragraph: ")

final_prompt=prompt.invoke(
    {"paragraph":para,
     'format_instructions': parser.get_format_instructions()}
)

response=model.invoke(final_prompt)
print(response.content)
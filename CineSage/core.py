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
     You are an expert movie analyst.

You will be given a single paragraph describing a movie.

Your task is to carefully read the paragraph and extract key information.

From the paragraph, identify and present:

- Movie Name
- Genre (if not explicitly mentioned, infer logically)
- Summary (rewrite in 3-4 lines clearly)
- Main Cast (only if mentioned in the paragraph)
- Key Themes or Concepts
- Notable Features (visuals, storytelling style, science, uniqueness)

Instructions:
- Input is just ONE paragraph, so extract carefully.
- Do NOT make up facts not present in the paragraph.
- If something is not mentioned (like cast), simply say "Not specified".
- Keep output clean, readable, and well-structured using headings or bullet points.
- Do NOT return JSON or any structured format.


"""),
    ("human",
     """
     this is the input paragraph
     Paragraph:
    {paragraph}
     """)
        
])

para=input("Give your Paragraph: ")

final_prompt=prompt.invoke(
    {"paragraph":para}
)

response=model.invoke(final_prompt)
print(response.content)
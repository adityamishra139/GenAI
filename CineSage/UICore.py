import streamlit as st
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser

# Load env
load_dotenv()

# Title
st.title("Movie Info Extractor")

# Input
paragraph = st.text_area("Enter Movie Paragraph:")

# Model
model = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0.9,
)

# Schema
class Movie(BaseModel):
    title: str
    release_year: Optional[int]
    genre: List[str]
    director: Optional[str]
    main_cast: List[str]
    rating: Optional[float]
    summary: str

parser = PydanticOutputParser(pydantic_object=Movie)

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system",
     """
extract movie information in the paragraph
{format_instructions}
"""),
    ("human",
     """
{paragraph}
""")
])

# Button
if st.button("Extract"):

    if paragraph.strip() == "":
        st.warning("Please enter a paragraph.")
    else:
        final_prompt = prompt.invoke({
            "paragraph": paragraph,
            "format_instructions": parser.get_format_instructions()
        })

        response = model.invoke(final_prompt)

        st.subheader("Output:")
        st.write(response.content)
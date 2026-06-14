from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatMessagePromptTemplate, ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_core.output_parsers import PydanticOutputParser
from collections import Counter 

load_dotenv()

model=ChatMistralAI(
    model="mistral-small-2506",
    temperature=0.7
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


prompts=ChatPromptTemplate.from_messages([
    ('system',
     """
     You are a helpful assistant and you need to extract movie information from the given paragraph.
     {movie_instructions}
     """),
    ('human',
     """
     extract movie information from the given paragraph.
     {paragraph}
     """)
    
])


data={
    "Interstellar is a visually stunning science fiction epic directed by Christopher Nolan that explores humanity’s struggle for survival beyond Earth. Set in a future where environmental collapse threatens extinction, the story follows Cooper, a former NASA pilot turned farmer, who is recruited for a daring mission through a wormhole near Saturn. The team travels across distant planets in search of a habitable world while grappling with time dilation caused by a massive black hole named Gargantua. The film blends emotional storytelling—particularly the bond between Cooper and his daughter Murph—with deep scientific concepts like relativity, gravity, and higher dimensions. Its climax ventures into a metaphysical space where love and time intersect, making it both intellectually stimulating and emotionally powerful.",
    "Inception, directed by Christopher Nolan, is a mind-bending thriller that delves into the architecture of dreams and the subconscious. The film follows Dom Cobb, a skilled thief who specializes in extracting secrets from people’s minds while they dream. He is offered a chance at redemption by performing the seemingly impossible task of “inception”—implanting an idea into someone’s subconscious. As Cobb and his team navigate multiple layers of dreams within dreams, the boundaries between reality and illusion blur. The narrative is complex and layered, with each dream level operating at a different time scale, culminating in an ambiguous ending that leaves audiences questioning the nature of reality itself.",
    "The Matrix, directed by The Wachowskis, is a groundbreaking sci-fi film that questions the nature of reality. It follows Neo, a computer hacker who discovers that the world he lives in is actually a simulated reality controlled by intelligent machines. Guided by Morpheus and Trinity, Neo learns about the Matrix and his role as “The One,” destined to challenge the system. The film combines philosophical ideas from simulation theory and existentialism with revolutionary visual effects like bullet time. Its narrative explores themes of free will, control, and awakening, making it one of the most influential sci-fi films of all time."
}

movies=[]

for lines in data:
    final_prompt=prompts.invoke({
        "paragraph":lines,
        "movie_instructions": parser.get_format_instructions()
    })
    response=model.invoke(final_prompt)
    try:
        parsed=parser.parse(response.content)
        movies.append(parsed)
    except Exception as e:
        print(f"Error parsing response: {e}")

all_genres=[]

for mo in movies:
    all_genres.extend(mo.genre)
genre_count=Counter(all_genres)

print("\n--- Most Common Genres ---")
print(genre_count)
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools.tools import tavily_search, scrape_website, extract_text
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# prompts

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. You are given a topic and you need to write a detailed research paper on the topic."),
    ("user", """Write a detailed research paper on the topic below.

    Topic: {topic}

    Research Gathered: {research}

    Structure the report as:
    - Introduction
    - Key Findings (minimum 3 well-explained points)
    - Conclusion
    - Sources (list all URLs found in the research)

    Be detailed, factual and professional."""),
])

critic_prompt = ChatPromptTemplate.from_messages(
    [("system","You are a sharp and constructive research critic. Be honest and specific."),
    ("user", """Evaluate the Research paper below and evaluate it strictly.

    Report: {report}

    Respond in this exact format.

    Score: X / 10

    Strengths:
    - .....
    - .....

    Areas to improve:
    - .....
    - .....

    One Line verdict:
    - ......
    
    """)]
)

# 1st Agent: Search Agent (uses tavily_search tool)
def build_search_agent():
    return create_agent(
        model=llm,
        tools=[tavily_search],
    )

# 2nd Agent: Scrape Agent (uses scrape_website and extract_text tools)
def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_website, extract_text],
        system_prompt=(
            "You are a research reader. Use only complete URLs copied exactly from "
            "the search results. Scrape the most relevant one. If the tool returns "
            "an HTTP error, try the next URL. Return the extracted page text and "
            "the source URL. Do not shorten, guess, or invent links."
        ),
    )

writer_chain = writer_prompt | llm | StrOutputParser()
critic_chain = critic_prompt | llm | StrOutputParser()
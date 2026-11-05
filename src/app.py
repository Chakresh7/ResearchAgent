from tools.tools import tavily_search, scrape_website, extract_text
from rich import print

result = tavily_search.invoke({"query": "What is the capital of France?"})

print(result)
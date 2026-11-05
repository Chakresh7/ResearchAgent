from agents.agent import build_search_agent, build_reader_agent, writer_chain, critic_chain

def run_research_agent(topic: str) -> dict:

    state = {}

    #step 1 - Search Agent

    print("\n"+" ="*50)
    print("STEP 1: Search Agent is working...")
    print("="*50)

    search_agent = build_search_agent()
    search_result = search_agent.invoke(
        {
            "messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]
        }
    )

    state['search_results'] = search_result['messages'][-1].content

    print("\n Search Result:",state['search_results'])

    #Step 2- Scrape Agent 
    print("\n"+" ="*50)
    print("STEP 2: Reader Agent is scraping top resources ...")
    print("="*50)

    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke({
        "messages": [("user",
                f"Based on the following search results about: '{topic}', "
                f"scrape the most relevant full URL for deeper content. "
                f"Copy the URL exactly as written. If that page returns an error, "
                f"try the next URL in the list.\n\n"
                f"Search Results:\n{state['search_results']}"
                )]
    })

    state['scraped_content'] = reader_result['messages'][-1].content

    print("\n Scraped Content:",state['scraped_content'])

    #step 3 - writer chain

    print("\n"+" ="*50)
    print("step 3 - Writer is drafting the report .")
    print("="*50)

    research_combined = (
        f"SEARCH RESULTS : \n {state['search_results' ]} \n\n"
        f"DETAILED SCRAPED CONTENT : \n {state[ 'scraped_content' ]}"
        )

    state["report"] = writer_chain.invoke({
            "topic" : topic,
            "research" : research_combined
            })

    print("\n Final Report\n",state['report' ])

    #critic report

    print("\n"+"="*50)
    print("step 4 - critic is reviewing the report ")
    print("="*50)

    state["feedback"] = critic_chain.invoke({
            "report":state['report']
    })

    print("\n critic report \n", state['feedback' ])

    return state
    
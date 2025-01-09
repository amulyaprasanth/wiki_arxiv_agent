from functools import lru_cache
import warnings
from langchain_ollama import ChatOllama, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper

@lru_cache(maxsize=128)
def get_wikipedia_query():
    return WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

@lru_cache(maxsize=128)
def get_arxiv_query():
    return ArxivQueryRun(api_wrapper=ArxivAPIWrapper())


# Create tools
wikipedia = get_wikipedia_query()
arxiv = get_arxiv_query()
tools = [wikipedia, arxiv]
# Initiate an llm
llm = ChatOllama(model="llama3.1", temperature=0.5, base_url="http://ollama-container:11434")

# Initialize the prompt
prompt = ChatPromptTemplate.from_messages(
        [
                ("system", "You are a helpful assistant. Answer in 1 or 2 paragraphs"),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
        ]
)

def wiki_arxiv_agent(query: str) -> str:
    """
    Creates and executes an agent that processes a query using Wikipedia and Arxiv tools.

    This function initializes a language model and a prompt template to create an agent
    capable of querying Wikipedia and Arxiv. It then executes the agent with the provided
    query and returns the result.

    Args:
        query (str): The input query to be processed by the agent.

    Returns:
        The result of the agent's execution on the given query.
    """

    # create an agent
    agent = create_tool_calling_agent(llm, tools, prompt)

    # Create agent executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    try:
        return agent_executor.invoke({"input": query})['output']
    except Exception as e:
        warnings.warn(f"An error occurred during agent execution: {e}")
        return None
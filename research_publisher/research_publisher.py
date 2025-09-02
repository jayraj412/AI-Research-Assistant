import os
import uuid
import datasets
from dotenv import load_dotenv

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools import PythonREPLTool
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain_chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredURLLoader


# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CX = os.getenv("GOOGLE_CX")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "llama3.2:1b")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:latest")

# Initialize LLM
chat_model = ChatOllama(
    model=OLLAMA_CHAT_MODEL,
    temperature=0
)

# Create embeddings
embedding_model = OllamaEmbeddings(
    model=OLLAMA_EMBEDDING_MODEL
)

# Initialize ChromaDB
PERSIST_DIRECTORY = "chroma_db"
vectorstore = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embedding_model
)

# Load and process documents for RAG
try:
    knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")
    knowledge_base = knowledge_base.filter(lambda row: row["source"].startswith("huggingface/transformers"))
    source_docs = [
        Document(page_content=doc["text"], metadata={"source": doc["source"].split("/")[1]})
        for doc in knowledge_base
    ]

    text_splitter = CharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )

    sample_urls = [
        "https://www.jioinstitute.edu.in/academics/artificial-intelligence-data-science"
    ]
    loader = UnstructuredURLLoader(urls=sample_urls)
    docs = loader.load()
    new_docs = text_splitter.split_documents(docs)

    num_docs = min(100, len(new_docs))
    uuids = [str(uuid.uuid4()) for _ in range(num_docs)]
    vectorstore.add_documents(
        documents=new_docs[:num_docs],
        ids=uuids
    )
except Exception as e:
    print(f"Error loading or processing RAG documents: {e}")
    # Handle the case where dataset loading fails, e.g., by skipping RAG agent or
    # providing a dummy vectorstore or a warning. For now, it will proceed
    # but rag_agent might fail if vectorstore is not properly initialized.

# Initialize tools
web_search_tool = TavilySearchResults(max_results=5)
python_repl_tool = PythonREPLTool()

# Define RAG Tool
def rag_search(query:str):
    "Function to do RAG search that returns 4 documents"
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.get_relevant_documents(query)
    return "\\nRetrieved documents:\\n" + "".join(
        [
            f"\\n\\n===== Document {str(i)} =====\\n" + doc.page_content
            for i, doc in enumerate(docs)
        ]
    )

# Define reviewer tool
def review_content(input:str):
    "Reviews the content"
    prompt = f"review the given content {input} , make necessary corrections and rewrite in a positive tone "
    return chat_model.invoke(prompt).content # Access content attribute

# Define writer tool
def writer(text_to_dump:str):
    "Writes in txt file"
    with open("final.txt", "w") as f:
        f.write(text_to_dump)
    return "Text has been written to final.txt" # Return a string to be compatible with tool output

# Create Specialized Agents
research_agent = create_react_agent(
    model=chat_model,
    tools=[web_search_tool, python_repl_tool],
    name="research_agent",
    prompt=(
        "You are an AI research assistant. "
        "You have two tools at your disposal:\\n"
        "  1) web_search — to search the web (max 5 results)\\n"
        "  2) python_repl — to execute Python code snippets safely\\n\\n"
        "Think through whether to call web_search or python_repl, "
        "use them as needed, then give the final answer."
    )
)

rag_agent = create_react_agent(
    chat_model,
    tools=[rag_search, python_repl_tool],
    name="rag_agent",
    prompt=(
        "You are a Retrieval-Augmented Generation (RAG) agent. "
        "You have one tool available:\\n"
        "  • rag_search(query: str) → str: retrieves the top 4 relevant document chunks.\\n\\n"
        "When given a user query, think step-by-step: decide whether to call rag_search, "
        "invoke it with the appropriate query, then use the returned documents to craft a complete answer."
    )
)

reviewing_agent = create_react_agent(
    chat_model,
    tools=[python_repl_tool, review_content], # Added review_content tool
    name="reviewing_agent",
    prompt=(
        "You are a Python code review assistant. "
        "You have two tools:\n"
        "  • python_repl(code: str) → str: execute Python code and return the output.\n"
        "  • review_content(input: str) → str: Reviews the given content, makes necessary corrections, and rewrites in a positive tone.\n\n"
        "When given content to review, think through whether to call review_content, "
        "invoke the tool to validate behavior, and then provide concise feedback or the reviewed content."
    )
)

writing_agent = create_react_agent(
    chat_model,
    tools=[rag_search, writer],
    name="writing_agent",
    prompt=(
        "You are a creative writing assistant. "
        "You have two tools available:\\n"
        "  • rag_search(query: str) → str: retrieves the top 4 relevant document chunks.\\n"
        "  • writer(text_to_dump: str) → str: Writes the given text to 'final.txt'.\\n\\n"
        "When given a topic or user prompt, think step-by-step: decide if you need to call rag_search, "
        "invoke it to gather background, then compose a clear, well-structured written response tailored to the user\'s needs. "
        "Finally, use the writer tool to save the generated content to a file if requested."
    )
)

visualization_agent = create_react_agent(
    chat_model,
    tools=[python_repl_tool],
    name="visualization_agent",
    prompt=(
        "You are a data visualization assistant. "
        "You have one tool available:\\n"
        "  • python_repl(code: str) → str: execute Python code and return the output.\\n\\n"
        "When given a request to visualize data, think through whether to call python_repl to generate charts, "
        "invoke it with the appropriate code (e.g., matplotlib), then present the final visualization or code snippet."
    )
)

# Create Supervisor Workflow
search_team = create_supervisor(
    agents = [research_agent, rag_agent],
    model=chat_model,
    prompt=(
        "You are a team supervisor managing a web search expert and a RAG search expert. "
        "For current events and information, use research_agent."
        "For transformer related information , use rag_agent."
    )
).compile(name="search_team")

publishing_team = create_supervisor(
    agents=[reviewing_agent, writing_agent, visualization_agent],
    model=chat_model,
    prompt=(
        "You are a publishing supervisor coordinating three specialists:\\n"
        "- reviewing_agent reviews and refines content.\\n"
        "- writing_agent crafts clear, engaging content using RAG for sourcing.\\n"
        "- visualization_agent generates charts and visual aids via Python REPL.\\n\\n"
        "For content review, delegate to reviewing_agent; for narrative and copy tasks, delegate to writing_agent; for any visual or chart needs, delegate to visualization_agent. "
        "Then combine their outputs into a cohesive deliverable."
    )
).compile(name="publishing_team")

top_level_supervisor = create_supervisor(
    agents=[search_team, publishing_team],
    model=chat_model,
    prompt=(
        "You are the top-level supervisor coordinating two teams:\\n"
        "- search_team: web research and retrieval experts\\n"
        "- publishing_team: content writing and visualization experts\\n\\n"
        "Analyze the user’s request, delegate subtasks to the appropriate team, "
        "and then merge their outputs into a single response."
    ),
    output_mode="full_history"
).compile(name="top_level_supervisor")


def run_research_publisher(query: str) -> str:
    """
    Run the research/publishing pipeline for the given query.
    """
    try:
        result = top_level_supervisor.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ]
        })
        # Extract the final message content from the full history
        final_message_content = ""
        # Iterate through messages in reverse to find the final, human-readable output
        for message in reversed(result['messages']):
            if hasattr(message, 'content') and message.content:
                # Prioritize messages from supervisor or assistant roles as final answers
                if (hasattr(message, 'name') and message.name == 'supervisor') or \
                   (hasattr(message, 'role') and message.role == 'assistant'):
                    final_message_content = message.content
                    break # Found the final answer, exit loop
                # Also consider tool outputs if they are the last substantial message
                elif hasattr(message, 'type') and message.type == 'tool_output':
                    final_message_content = message.content
                    # Don't break immediately, in case a supervisor message comes after a tool output

        return final_message_content if final_message_content else str(result)
    except Exception as e:
        return f"An error occurred during multi-agent pipeline execution: {e}"

if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv

    load_dotenv()

    if len(sys.argv) < 2:
        print("Usage: python research_publisher.py \"your query\"")
        sys.exit(1)

    q = " ".join(sys.argv[1:])
    print(run_research_publisher(q))



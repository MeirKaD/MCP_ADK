import os
import threading
import asyncio
from dotenv import load_dotenv
from google.adk.agents import Agent, SequentialAgent, LoopAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.events import Event, EventActions
from google.genai import types
from typing import Optional

load_dotenv()

_mcp_tools = None
_exit_stack = None
_initialized = False
_initialization_in_progress = False
_init_lock = threading.Lock()

print("Module loaded: web_research_agent")

def create_planner_agent():
    return Agent(
        name="planner",
        model="gemini-2.0-flash",
        description="Plans research by breaking down complex topics into search queries",
        instruction="""
        You are a research planning expert. Your task is to:
        1. Analyze the user's research topic
        2. Break it down into 3-5 specific search queries that together will cover the topic comprehensively
        3. Output a JSON object with format: {"queries": ["query1", "query2", "query3"]}
        Be concise and focused in your search queries.
        """,
        output_key="search_queries"
    )

# Define researcher agent with improved tool guidance
def create_researcher_agent():
    return Agent(
        name="researcher",
        model="gemini-2.0-flash",
        description="Executes web searches and extracts relevant information",
        instruction="""
        You are a web researcher. You will:
        1. Take the specific search queries from the planner
        2. For EACH query:
           a. Use search_engine to find relevant information (start with "google" engine)
           b. Select 2-3 most relevant results and for each result:
              i. Use scraping_browser_navigate to navigate to the URL
              ii. Use scraping_browser_get_text to extract the main content
              iii. If needed, use scraping_browser_links to find important sections and scraping_browser_click to navigate to them
           c. If a page fails to load or lacks information, try another result
        3. Summarize key findings for each query with source citations
        
        IMPORTANT: 
        - Always begin with search_engine to discover relevant pages
        - Then use browser tools in this sequence:
          1. scraping_browser_navigate (to go to the URL)
          2. scraping_browser_get_text (to extract content)
          3. scraping_browser_links and scraping_browser_click (if you need to navigate within the site)
        - Include clear citations with URLs for each piece of information
        - Format your findings for each search query separately
        """,
        before_model_callback=check_researcher_tools
    )

# Define publisher agent with clear instruction
def create_publisher_agent():
    return Agent(
        name="publisher",
        model="gemini-2.0-flash", 
        description="Synthesizes research findings into a comprehensive and detailed final document",
        instruction="""
        You are an expert Technical Writer and Synthesist. Your mission is to transform the detailed research findings provided by the researcher into a comprehensive, well-structured, and insightful final report.

        Follow these steps meticulously:
        1.  **Deep Analysis & Synthesis:** Carefully review *all* the research findings, summaries, and cited sources provided by the researcher for *all* search queries. Do not just list findings; **synthesize** them. Identify connections, relationships, common themes, contrasting points, and overall patterns across the different pieces of information and sources.
        2.  **Logical Structure:** Organize the synthesized information into a coherent and deeply structured document. Use logical sections and sub-sections with clear, descriptive headings (using Markdown H2, H3, etc.) to group related concepts and findings. A possible structure could be: Introduction, Key Theme/Aspect 1 (with sub-points), Key Theme/Aspect 2 (with sub-points), ..., Conclusion, References. Adapt the structure based on the content.
        3.  **Compelling Introduction:** Write a robust introduction that clearly defines the topic, states the report's main objectives, highlights the key questions or areas explored, and provides a roadmap for the reader, outlining the main sections of the report.
        4.  **Detailed Body Sections:** Elaborate on the synthesized findings within each section. Provide sufficient detail and explanation. Explain concepts clearly. Ensure that claims and statements are directly supported by the research gathered by the researcher. **Explicitly reference the source URLs** where appropriate within the text (e.g., "According to [Source URL], ..."). Aim for thoroughness and depth, ensuring all significant aspects uncovered by the research are included. Use bullet points or numbered lists for clarity where appropriate. Ensure smooth transitions between paragraphs and sections.
        5.  **Insightful Conclusion:** Craft a strong conclusion that summarizes the most important findings and synthesized insights from the report. Briefly reiterate the main points discussed. You may also briefly mention limitations based *only* on the provided research or suggest natural next steps *if strongly implied* by the findings, but do *not* introduce entirely new information or opinions.
        6.  **Professional Formatting:** Format the entire document using clean and consistent Markdown. Utilize headings, lists (bulleted and numbered), bold/italic emphasis, and potentially blockquotes effectively to enhance readability and structure.
        7.  **Comprehensive References:** Create a dedicated "References" section at the very end. List *all* unique source URLs that were cited in the researcher's findings and used in your report. Ensure the list is clean and easy to read.
        8.  **Tone and Quality:** Maintain a professional, objective, and informative tone throughout the report. Ensure the language is clear, precise, and accurate according to the research. Strive for a high-quality, polished final document that is significantly more detailed and synthesized than the raw researcher output. Cover all key aspects comprehensively.
        """,
        output_key="final_document"
    )

# Create a single initialization function that leverages the EXISTING event loop
async def initialize_mcp_tools():
    """Initialize MCP tools using the existing event loop."""
    global _mcp_tools, _exit_stack, _initialized, _initialization_in_progress
    
    if _initialized:
        return _mcp_tools
    
    with _init_lock:
        if _initialized:
            return _mcp_tools
            
        if _initialization_in_progress:
            while _initialization_in_progress:
                await asyncio.sleep(0.1)
            return _mcp_tools
        
        _initialization_in_progress = True
    
    try:
        from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
        
        print("Connecting to Bright Data MCP...")
        tools, exit_stack = await MCPToolset.from_server(
            connection_params=StdioServerParameters(
                command='npx',
                args=["-y", "@brightdata/mcp"],
                env={
                    "API_TOKEN": "YOUR_API_TOKEN",
                    "WEB_UNLOCKER_ZONE": "UB_ZONE",
                    "BROWSER_AUTH": "SBR_USER:SBR_PASS"
                }
            )
        )
        print(f"MCP Toolset created successfully with {len(tools)} tools")
        
        _mcp_tools = tools
        _exit_stack = exit_stack
        
        import atexit
        
        def cleanup_mcp():
            global _exit_stack
            if _exit_stack:
                print("Closing MCP server connection...")
                try:

                    loop = asyncio.new_event_loop()
                    loop.run_until_complete(_exit_stack.aclose())
                    loop.close()
                    print("MCP server connection closed successfully.")
                except Exception as e:
                    print(f"Error closing MCP connection: {e}")
                finally:
                    _exit_stack = None
        
        atexit.register(cleanup_mcp)
        
        _initialized = True
        
        # Find and update the researcher agent if root_agent is defined
        for agent in root_agent.sub_agents:
            if agent.name == "researcher":
                agent.tools = tools
                print(f"Successfully added {len(tools)} tools to researcher agent")
                
                # List some tool names for debugging
                tool_names = [tool.name for tool in tools[:5]]
                print(f"Available tools include: {', '.join(tool_names)}")
                break
                
        print("MCP initialization complete!")
        return tools
        
    except Exception as e:
        print(f"Error initializing MCP tools: {e}")
        return None
    finally:
        _initialization_in_progress = False


async def wait_for_initialization():
    """Wait for MCP initialization to complete."""
    global _initialized
    
    if not _initialized:
        print("Starting initialization in callback...")
        await initialize_mcp_tools()
    
    return _initialized

def check_researcher_tools(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
    global _mcp_tools, _initialized
    
    agent_name = callback_context.agent_name
    
    if agent_name == "researcher" and not _initialized:
        print("Researcher agent needs tools - will start initialization")
        
        loop = asyncio.get_event_loop()
        loop.create_task(initialize_mcp_tools())
        
        print("Initialization started in background. Asking user to retry.")
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text="Initializing research tools. This happens only once. Please try your query again in a few moments.")]
            )
        )
    
    return None

root_agent = SequentialAgent(
    name="web_research_agent",
    description="An agent that researches topics on the web and creates comprehensive reports",
    sub_agents=[
        create_planner_agent(),
        create_researcher_agent(),
        create_publisher_agent()
    ]
)

print("Agent structure created. MCP tools will be initialized on first use.")

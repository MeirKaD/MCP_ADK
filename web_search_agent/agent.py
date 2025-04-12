import os
import threading
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext

load_dotenv()

root_agent = Agent(
    name="websearch_assistant",  
    model="gemini-2.0-flash",
    description="Agent to help search the web and retrieve information",  
    instruction="I can help you search the web for information. You can ask me to search for topics, retrieve data from websites, and find answers to your questions.", 
    tools=[], 
)

_exit_stack = None
_initialized = False
_init_lock = threading.Lock()

def before_agent_callback(callback_context: CallbackContext):
    """This callback initializes MCP tools just before the agent runs."""
    global _exit_stack, _initialized

    
    if not _initialized:
        with _init_lock:
            if not _initialized:
                import asyncio
                
                async def initialize_mcp():
                    """Initialize MCP tools asynchronously."""
                    global _exit_stack, _initialized
                    try:
                        from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
                        
                        
                        print(f"Connecting to Bright Data MCP")
                        tools, exit_stack = await MCPToolset.from_server(
                            connection_params=StdioServerParameters(
                                command='npx',
                                args=["-y", 
                                      "@brightdata/mcp"],
                                      env={
                                          "API_TOKEN": "<API_TOKEN>",
                                 "WEB_UNLOCKER_ZONE": "unblocker",
                                "BROWSER_AUTH": "brd-customer-<ID>-zone-scraping_browser:<PASS>"
                                      }
                            )
                        )
                        print(f"MCP Toolset created with {len(tools)} tools.")
                        
                        _exit_stack = exit_stack
                        
                        root_agent.tools = tools
                        
                        import atexit
                        
                        def cleanup_mcp():
                            """Clean up MCP resources when the process exits."""
                            global _exit_stack
                            if _exit_stack:
                                print("Closing MCP server connection...")
                                loop = asyncio.new_event_loop()
                                try:
                                    loop.run_until_complete(_exit_stack.aclose())
                                    print("MCP server connection closed successfully.")
                                except Exception as e:
                                    print(f"Error closing MCP connection: {e}")
                                finally:
                                    loop.close()
                                    _exit_stack = None
                        
                        atexit.register(cleanup_mcp)
                        
                        _initialized = True
                        print("MCP initialization complete!")
                    except Exception as e:
                        print(f"Error initializing MCP tools: {e}")
                
                loop = asyncio.get_event_loop()
                loop.create_task(initialize_mcp())
    
    return None

root_agent.before_agent_callback = before_agent_callback

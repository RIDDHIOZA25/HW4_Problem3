"""
ReAct Agent Implementation with Search, Compare, and Analyze Tools
Adapted from notebook for local execution
"""
import os
from dotenv import load_dotenv
from typing import List, Tuple, Any, Optional, Callable
import time

# Load environment variables
load_dotenv()

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish

# Try different import paths for agent creation
try:
    # Try langchain_classic which has the legacy API
    from langchain_classic.agents import initialize_agent, AgentType
    create_react_agent = None
except ImportError:
    try:
        # Try LangChain 1.0+ API
        from langchain.agents import AgentExecutor
        from langchain.agents.react.agent import create_react_agent
    except ImportError:
        try:
            # Try alternative import path
            from langchain.agents import AgentExecutor, create_react_agent
        except ImportError:
            # Last resort: try experimental
            from langchain_experimental.agents import initialize_agent, AgentType
            create_react_agent = None

# Set API keys from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY not found in environment variables")

# Initialize Gemini model with rate limiting
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    temperature=0,
    google_api_key=GOOGLE_API_KEY,
    max_retries=5,  # Retry on rate limit errors
    request_timeout=120  # Increase timeout for rate-limited requests
)

# ========== a) Search Tool ==========
def formatted_search_results(query: str) -> str:
    """
    Calls Tavily and returns a clean, readable string of search results.
    The output is always a string, which ensures ReAct can process it.
    """
    time.sleep(1)  # Delay to avoid rate limiting
    raw_results = TavilySearchResults(k=5, tavily_api_key=TAVILY_API_KEY).run(query)

    # Format list output into a readable string
    if isinstance(raw_results, list):
        formatted = []
        for i, r in enumerate(raw_results):
            content = r.get("content", "")
            url = r.get("url", "")
            formatted.append(f"Result {i+1}:\n{content[:1000]}\nSource: {url}\n")
        return "\n\n".join(formatted)

    return str(raw_results)

search_tool = Tool(
    name="Search",
    func=formatted_search_results,
    description="Useful for searching the web for relevant information. Input should be a search query."
)

# ========== b) Comparison Tool ==========
def compare_items(query: str) -> str:
    """
    Takes a comma-separated string with multiple items and a category.
    Returns a comparison across those items using Gemini.
    Example input: "iPhone 15 Pro, Samsung Galaxy S23 Ultra, Google Pixel 8, smartphone"
    """
    try:
        # Step 1: Parse items and category
        parts = [p.strip() for p in query.split(",")]
        if len(parts) < 3:
            return "Please provide at least two items and a category, separated by commas."

        items = parts[:-1]  # all except last
        category = parts[-1]

        # Step 2: Build comparison prompt
        prompt_template = PromptTemplate(
            input_variables=["items", "category"],
            template="""
Compare the following {category}s:
{items}

Focus on key features, advantages, and any major differences.
Be concise but informative.
"""
        )

        # Step 3: Run Gemini comparison directly
        time.sleep(2)  # Delay to avoid rate limiting
        formatted_prompt = prompt_template.format(items=', '.join(items), category=category)
        response = llm.invoke(formatted_prompt)
        return response.content if hasattr(response, 'content') else str(response)

    except Exception as e:
        return f"An error occurred in the compare tool: {str(e)}"

compare_tool = Tool(
    name="Compare",
    func=compare_items,
    description="Compares multiple items in a category. Input format: item1, item2, ..., category"
)

# ========== c) Analysis Tool ==========
def analyze_results(results: str, query: str) -> str:
    prompt_template = PromptTemplate(
        input_variables=["results", "query"],
        template="""
You are an assistant analyzing the following content in response to a user question.

User Query:
{query}

Content to Analyze:
{results}

Instructions:
- Summarize the most important points.
- Highlight any clear advantages, disadvantages, or insights.
- Be concise and relevant to the user's question.
- Avoid repeating too much raw content.
- Return ONLY the analysis text, no additional formatting or metadata.
"""
    )
    # Use LLM directly instead of LLMChain
    time.sleep(2)  # Delay to avoid rate limiting
    formatted_prompt = prompt_template.format(results=results, query=query)
    response = llm.invoke(formatted_prompt)
    
    # Extract text content properly - handle various response formats
    try:
        if hasattr(response, 'content'):
            content = response.content
            # If content is a list, extract text from it
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict):
                        if 'text' in item:
                            text_parts.append(str(item['text']))
                        elif 'type' in item and item.get('type') == 'text' and 'text' in item:
                            text_parts.append(str(item['text']))
                    elif isinstance(item, str):
                        text_parts.append(item)
                result = '\n'.join(text_parts) if text_parts else str(response)
                # Clean up any remaining dict/json artifacts
                if isinstance(result, str) and result.startswith('{'):
                    try:
                        import json
                        parsed = json.loads(result)
                        if isinstance(parsed, dict) and 'text' in parsed:
                            return str(parsed['text'])
                    except:
                        pass
                return result
            # If content is a string, return it
            elif isinstance(content, str):
                return content
            # Otherwise convert to string
            else:
                return str(content)
        elif isinstance(response, dict):
            # Handle dict response format
            if 'text' in response:
                return str(response['text'])
            elif 'content' in response:
                return str(response['content'])
            elif 'type' in response and response.get('type') == 'text' and 'text' in response:
                return str(response['text'])
        # Last resort: convert to string and try to extract text
        response_str = str(response)
        if 'text' in response_str and response_str.startswith('{'):
            try:
                import json
                parsed = json.loads(response_str)
                if isinstance(parsed, dict) and 'text' in parsed:
                    return str(parsed['text'])
            except:
                pass
        return response_str
    except Exception as e:
        # Fallback: return string representation
        return str(response)

# Track if Analyze has been called to prevent loops
_analyze_called = False

def analyze_wrapper(full_input: str) -> str:
    global _analyze_called
    if _analyze_called:
        return "Analyze tool has already been used. Please provide a Final Answer now."
    _analyze_called = True
    
    try:
        # Simple parsing - handle newlines
        if "\n\n" in full_input:
            query, results = full_input.split("\n\n", 1)
        elif "\\n\\n" in full_input:
            query, results = full_input.split("\\n\\n", 1)
            results = results.replace("\\n", "\n")
        elif "\n" in full_input:
            parts = full_input.split("\n", 1)
            query, results = (parts[0], parts[1]) if len(parts) == 2 else (full_input, "")
        else:
            # Split on first period
            idx = full_input.find(".")
            if idx > 0:
                query = full_input[:idx+1]
                results = full_input[idx+1:].strip()
            else:
                return f"Format: <query>\\n\\n<results>. Received: {full_input[:200]}"
        
        query = query.strip()
        results = results.strip()
        
        if not query or not results:
            return f"Missing query or results. Query: {query[:50]}, Results: {results[:50]}"
        
        return analyze_results(results=results, query=query)
    except Exception as e:
        _analyze_called = False  # Reset on error
        return f"Error: {str(e)}"

analyze_tool = Tool(
    name="Analyze",
    func=analyze_wrapper,
    description=(
        "Use this tool ONCE to analyze and summarize search results or comparisons. "
        "Input format: <original user query>\\n\\n<results>. "
        "After using this tool, you MUST provide a Final Answer. Do NOT use this tool again."
    )
)

# ========== d) ReAct Agent Integration ==========
react_format_instruction = """
IMPORTANT RULES:
1. Use the Analyze tool ONLY ONCE per query - do not repeat it
2. After using Analyze, you MUST provide a Final Answer immediately
3. Format Analyze input as: <query>\\n\\n<results>
4. Do NOT call Analyze multiple times with the same data
"""

# Gather all tools
tools = [search_tool, compare_tool, analyze_tool]

# Create ReAct agent - try multiple approaches
if create_react_agent is not None:
    # Use new LangChain 1.0+ API
    try:
        from langchain import hub
        try:
            prompt = hub.pull("hwchase17/react")
        except:
            # Fallback: create a ReAct prompt template
            prompt = PromptTemplate.from_template("""You are a helpful assistant that can use tools to answer questions.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

{agent_scratchpad}

Additional Instructions:
{react_format_instruction}
""")
        
        # Add the custom instruction to the prompt
        prompt = prompt.partial(react_format_instruction=react_format_instruction)
        
        # Create the agent
        agent = create_react_agent(llm, tools, prompt)
        
        # Create the agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            return_intermediate_steps=True,
            handle_parsing_errors="Check your output and make sure it follows the format. Provide a Final Answer after using tools.",
            max_iterations=10  # Strict limit to prevent loops
        )
    except Exception as e:
        # If new API fails, fall back to legacy
        print(f"Warning: Could not use new API: {e}. Falling back to legacy API.")
        create_react_agent = None

if create_react_agent is None:
    # Use legacy initialize_agent API from langchain_classic
    agent_executor = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        agent_kwargs={"system_message": react_format_instruction},
        return_intermediate_steps=True,
        max_iterations=10,  # Strict limit to prevent loops
        handle_parsing_errors="Check your output and make sure it follows the format. Provide a Final Answer after using tools."
    )

def process_query(query: str, max_steps: int = 100, step_callback: Optional[Callable] = None):
    """
    Process a query using the ReAct agent and return both the result and intermediate steps.
    
    Args:
        query: The user's query
        max_steps: Maximum number of agent steps
        step_callback: Optional callback function to call with (step_type, data) for real-time updates
    """
    global _analyze_called
    _analyze_called = False  # Reset for each query
    
    # List to store intermediate steps
    intermediate_steps: List[Tuple[AgentAction, str]] = []
    thoughts: List[str] = []
    
    # Custom callback handler to capture intermediate steps
    class StepCaptureHandler(BaseCallbackHandler):
        def __init__(self, steps_list, thoughts_list, callback=None):
            super().__init__()
            self.steps_list = steps_list
            self.thoughts_list = thoughts_list
            self.callback = callback
        
        def on_agent_action(self, action: AgentAction, **kwargs: Any) -> None:
            self.steps_list.append((action, ""))
            if self.callback:
                self.callback("action", {
                    "tool": action.tool if hasattr(action, 'tool') else str(action),
                    "input": action.tool_input if hasattr(action, 'tool_input') else ""
                })
        
        def on_tool_end(self, output: str, **kwargs: Any) -> None:
            if self.steps_list:
                # Update the last step with the observation
                last_action, _ = self.steps_list[-1]
                # Clean up output if it's a complex object
                clean_output = output
                if isinstance(output, dict):
                    if 'text' in output:
                        clean_output = output['text']
                    elif 'content' in output:
                        clean_output = str(output['content'])
                    else:
                        clean_output = str(output)
                elif isinstance(output, list):
                    # Extract text from list of dicts
                    text_parts = []
                    for item in output:
                        if isinstance(item, dict) and 'text' in item:
                            text_parts.append(item['text'])
                        elif isinstance(item, str):
                            text_parts.append(item)
                    clean_output = '\n'.join(text_parts) if text_parts else str(output)
                
                self.steps_list[-1] = (last_action, clean_output)
                if self.callback:
                    self.callback("observation", {"output": str(clean_output)[:500]})  # Truncate for display
        
        def on_llm_start(self, serialized: dict, prompts: List[str], **kwargs: Any) -> None:
            # Extract thought from prompt if possible
            if prompts and self.callback:
                prompt_text = prompts[0] if prompts else ""
                if "Thought:" in prompt_text:
                    thought = prompt_text.split("Thought:")[-1].split("Action:")[0].strip()
                    if thought:
                        self.thoughts_list.append(thought)
                        self.callback("thought", {"thought": thought})
    
    try:
        # Run the agent with callback handler
        handler = StepCaptureHandler(intermediate_steps, thoughts, step_callback)
        # Update max_iterations for this query
        if hasattr(agent_executor, 'max_iterations'):
            agent_executor.max_iterations = max_steps
        
        # Add delay between API calls to avoid rate limiting
        time.sleep(1)  # Initial delay
        
        result = agent_executor.invoke(
            {"input": query},
            config={"callbacks": [handler]}
        )
        
        # Add delay after completion
        time.sleep(0.5)
        
        # Get intermediate steps
        result_steps = result.get("intermediate_steps", [])
        if result_steps:
            intermediate_steps = result_steps
        
        # Get output - simple extraction
        output = result.get("output", "")
        
        # Clean complex objects
        if isinstance(output, dict):
            output = output.get('text', output.get('content', str(output)))
        elif isinstance(output, list):
            output = '\n'.join([item.get('text', str(item)) if isinstance(item, dict) else str(item) for item in output])
        
        # If no output, use last Analyze result
        if not output or len(str(output)) < 50:
            for step in reversed(intermediate_steps):
                if isinstance(step, tuple) and len(step) == 2:
                    action, obs = step
                    if hasattr(action, 'tool') and action.tool == "Analyze":
                        if isinstance(obs, dict) and 'text' in obs:
                            output = str(obs['text'])
                        elif isinstance(obs, list):
                            output = '\n'.join([item.get('text', str(item)) if isinstance(item, dict) else str(item) for item in obs])
                        else:
                            output = str(obs)
                        break
        
        return {
            "output": str(output) if output else "Please check the reasoning steps above for the answer.",
            "intermediate_steps": intermediate_steps
        }
    except RecursionError:
        return {
            "output": "The query was too complex and exceeded the maximum number of steps. Please try a simpler query.",
            "intermediate_steps": []
        }
    except Exception as e:
        error_msg = str(e)
        # Check if it's a parsing error and extract the actual answer if available
        if "Could not parse LLM output" in error_msg:
            # Try to extract the actual answer from the error message
            if "This is the error:" in error_msg:
                parts = error_msg.split("This is the error:")
                if len(parts) > 1:
                    actual_output = parts[1].split("For troubleshooting")[0].strip()
                    return {
                        "output": actual_output,
                        "intermediate_steps": intermediate_steps
                    }
        
        return {
            "output": f"An error occurred: {error_msg}",
            "intermediate_steps": intermediate_steps
        }


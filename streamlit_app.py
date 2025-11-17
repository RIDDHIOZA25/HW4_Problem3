"""
Streamlit UI for ReAct Agent with Multiple Tools
"""
import streamlit as st
from react_agent import process_query
import time

# Page configuration
st.set_page_config(
    page_title="ReAct Agent - Multiple Tools",
    page_icon="🤖",
    layout="wide"
)

# Title and description
st.title("🤖 ReAct Agent with Multiple Tools")
st.markdown("""
This agent can use three tools to answer your queries:
- **Search**: Search the web for information
- **Compare**: Compare multiple items in a category
- **Analyze**: Analyze and summarize results

Ask complex questions that require multiple tools!
""")

# Initialize session state
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "current_result" not in st.session_state:
    st.session_state.current_result = None
if "current_steps" not in st.session_state:
    st.session_state.current_steps = []
if "current_thoughts" not in st.session_state:
    st.session_state.current_thoughts = []
if "processing" not in st.session_state:
    st.session_state.processing = False
if "realtime_steps" not in st.session_state:
    st.session_state.realtime_steps = []

# Sidebar for example queries
with st.sidebar:
    st.header("📝 Example Queries")
    
    example_queries = [
        "Best universities in the US for Data Science",
        "Compare iPhone 15 vs Samsung S23",
        "Best programming languages for AI",
        "Top travel destinations in 2025"
    ]
    
    for i, example in enumerate(example_queries):
        if st.button(example, key=f"example_{i}", use_container_width=True):
            st.session_state.example_query = example
            st.rerun()

# Main input area
st.header("💬 Enter Your Query")

# Get initial value from example query or empty
initial_value = st.session_state.get("example_query", "")
user_query = st.text_area(
    "Your question:",
    value=initial_value,
    height=100,
    key="query_input"
)

# Clear example query after it's been set
if initial_value:
    st.session_state.example_query = ""

# Submit button
col1, col2 = st.columns([1, 5])
with col1:
    submit_button = st.button("🚀 Submit Query", type="primary", use_container_width=True)

# Process query when submitted
if submit_button and user_query:
    st.session_state.processing = True
    st.session_state.realtime_steps = []
    st.session_state.current_thoughts = []
    
    # Show processing status with progress
    with st.status("🤔 Agent is thinking and using tools...", expanded=True) as status:
        # Callback function to collect steps (will be called during processing)
        collected_steps = []
        collected_thoughts = []
        
        def update_step(step_type, data):
            if step_type == "thought":
                collected_thoughts.append(data["thought"])
                st.write(f"💭 **Thought {len(collected_thoughts)}:** {data['thought']}")
            elif step_type == "action":
                collected_steps.append({
                    "type": "action",
                    "tool": data.get("tool", ""),
                    "input": data.get("input", "")
                })
                st.write(f"🔧 **Action {len(collected_steps)}:** {data.get('tool', '')}[{data.get('input', '')}]")
            elif step_type == "observation":
                if collected_steps:
                    collected_steps[-1]["observation"] = data.get("output", "")
                    obs_preview = data.get("output", "")[:200] + "..." if len(data.get("output", "")) > 200 else data.get("output", "")
                    st.write(f"📊 **Observation:** {obs_preview}")
        
        # Process the query
        try:
            result = process_query(user_query, step_callback=update_step)
            
            # Store results in session state
            st.session_state.current_result = result["output"]
            st.session_state.current_steps = result["intermediate_steps"]
            st.session_state.query_history.append({
                "query": user_query,
                "result": result["output"],
                "steps": result["intermediate_steps"]
            })
            st.session_state.processing = False
            
            status.update(label="✅ Query completed!", state="complete")
            time.sleep(0.5)
            
        except Exception as e:
            status.update(label=f"❌ Error: {str(e)}", state="error")
            st.session_state.processing = False
    
    # Scroll to results
    st.rerun()

# Display results if available
if st.session_state.current_result:
    st.header("📊 Final Answer")
    st.markdown("---")
    st.markdown(st.session_state.current_result)
    
    # Display reasoning process
    if st.session_state.current_steps:
        st.header("🔍 Step-by-Step Reasoning Process")
        st.markdown("---")
        
        # Create expandable sections for each step
        for step_idx, step_data in enumerate(st.session_state.current_steps, 1):
            # Handle both tuple format (action, observation) and other formats
            if isinstance(step_data, tuple) and len(step_data) == 2:
                action, observation = step_data
                tool_name = action.tool if hasattr(action, 'tool') else str(action)
                tool_input = action.tool_input if hasattr(action, 'tool_input') else ""
                
                with st.expander(f"Step {step_idx}: {tool_name}", expanded=True):
                    st.markdown("**Action:**")
                    st.code(f"{tool_name}[{tool_input}]", language="text")
                    
                    st.markdown("**Observation:**")
                    # Truncate very long observations
                    obs_text = str(observation)
                    if len(obs_text) > 3000:
                        st.text_area(
                            "Observation (truncated):",
                            value=obs_text[:3000] + "\n\n... (truncated for display)",
                            height=250,
                            disabled=True,
                            key=f"obs_{step_idx}"
                        )
                    else:
                        st.text_area(
                            "Observation:",
                            value=obs_text,
                            height=min(250, max(100, len(obs_text) // 15)),
                            disabled=True,
                            key=f"obs_{step_idx}"
                        )
            else:
                # Fallback for other formats
                with st.expander(f"Step {step_idx}", expanded=True):
                    st.json(step_data)

# Display query history
if st.session_state.query_history:
    st.sidebar.header("📜 Query History")
    for idx, item in enumerate(reversed(st.session_state.query_history[-5:])):  # Show last 5
        with st.sidebar.expander(f"Query {len(st.session_state.query_history) - idx}"):
            st.text(item["query"][:100] + "..." if len(item["query"]) > 100 else item["query"])
            if st.button("Load", key=f"load_{idx}"):
                st.session_state.current_result = item["result"]
                st.session_state.current_steps = item["steps"]
                st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>ReAct Agent Implementation | Powered by LangChain & Gemini</p>
</div>
""", unsafe_allow_html=True)


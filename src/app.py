import streamlit as st
import os
import sys

# Ensure correct path mapping for local module discovery
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.dirname(current_dir))

from graphs.workflow import research_agent_graph

# Configure basic page layout
st.set_page_config(page_title="Advanced Research Agent", layout="wide")

st.title("Advanced Multi-Agent Research System")
st.caption("Automated market research platform utilizing Multi-Agent RAG and LangGraph orchestration")

# Initialize persistent session states for managing memory threads and conversation logs
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "streamlit_session_001"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "graph_state" not in st.session_state:
    # Initialize default graph state schema
    st.session_state.graph_state = {
        "user_query": "",
        "research_plan": [],
        "executed_queries": [],
        "raw_web_data": [],
        "financial_data": [],
        "vision_data": [],  # Holds the paths of the uploaded diagrams/images
        "refined_contexts": [],
        "current_report": "",
        "editor_feedback": {},
        "next_step": ""
    }

config = {"configurable": {"thread_id": st.session_state.thread_id}}

# Sidebar layout for asset uploads (Diagrams, charts, or maps)
with st.sidebar:
    st.header("Research Assets Ingestion")
    st.subheader("Multi-Modal Stream")
    uploaded_file = st.file_uploader(
        "Upload market charts or supply chain diagrams", 
        type=["png", "jpg", "jpeg"]
    )
    
    if uploaded_file:
        # Define strict local storage directory
        save_dir = os.path.join(os.path.dirname(current_dir), "data", "raw", "multimodal")
        os.makedirs(save_dir, exist_ok=True)
        
        # Save file locally to preserve image path infrastructure
        file_path = os.path.join(save_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"Ingested asset: {uploaded_file.name}")
        st.image(uploaded_file, caption="Target Research Diagram")
    else:
        file_path = None

# Render historical chat sequences to the user interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle real-time user chat inputs
if user_input := st.chat_input("Enter your research request (e.g., Analyze Vietnam EV market trends for 2026)..."):
    
    # Append and render user query on the dashboard
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
    # Trigger the Multi-Agent graph execution pipeline
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        report_placeholder = st.empty()
        
        with st.spinner("Agent workflow is processing..."):
            # Update internal state with new user criteria
            st.session_state.graph_state["user_query"] = user_input
            st.session_state.graph_state["next_step"] = "" 
            
            # Synchronize multi-modal files to graph state if present
            if file_path:
                st.session_state.graph_state["vision_data"] = [file_path]
            else:
                st.session_state.graph_state["vision_data"] = []
            
            # Invoke LangGraph workflow with thread configuration for memory persistence
            final_state = research_agent_graph.invoke(st.session_state.graph_state, config=config)
            
            # Persist updated state parameters back into session memory
            st.session_state.graph_state = final_state
            
            # Extract final synthesized compilation from the graph state
            generated_report = final_state.get("current_report", "Failed to generate the research report.")
            
        # Render the final Markdown report text on the web interface
        report_placeholder.markdown(generated_report)
        st.session_state.messages.append({"role": "assistant", "content": generated_report})
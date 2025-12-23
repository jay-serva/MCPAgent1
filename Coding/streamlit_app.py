import streamlit as st
import json
from pathlib import Path
from datetime import datetime
from agent import run_query, create_artifact_file, log_trace, export_graph_png

st.set_page_config(
    page_title="Know the Rules",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        border-bottom: 3px solid #1f4e79;
    }
    .chat-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #1f4e79;
        color: white;
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        text-align: right;
    }
    .bot-message {
        background-color: #e8eef4;
        color: #1f4e79;
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #1f4e79;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f4e79;
        color: white;
        font-weight: 600;
        border-radius: 5px;
        padding: 0.5rem;
    }
    .stButton>button:hover {
        background-color: #2e6da4;
    }
    .restart-button>button {
        background-color: #6c757d;
    }
    .restart-button>button:hover {
        background-color: #5a6268;
    }
    </style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.artifact_file = create_artifact_file()
    st.session_state.session_active = True
    st.session_state.artifact_ready = False

st.markdown('<div class="main-header">📋 Know the Rules</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Session Controls")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Restart", key="restart"):
            st.session_state.messages = []
            st.session_state.artifact_file = create_artifact_file()
            st.session_state.session_active = True
            st.session_state.artifact_ready = False
            st.experimental_rerun()
    
    with col2:
        if st.button("🛑 End", key="end"):
            st.session_state.session_active = False
            
            if st.session_state.artifact_file.exists():
                with open(st.session_state.artifact_file, 'r') as f:
                    artifact_data = json.load(f)
                
                artifact_data["session_end"] = datetime.now().isoformat()
                
                with open(st.session_state.artifact_file, 'w') as f:
                    json.dump(artifact_data, f, indent=2)
                
                st.session_state.artifact_ready = True
                st.experimental_rerun()
    
    st.markdown("---")
    st.markdown("### Session Info")
    if "artifact_file" in st.session_state:
        st.info(f"Artifact: `{st.session_state.artifact_file.name}`")
    if "messages" in st.session_state:
        st.info(f"Queries: {len(st.session_state.messages) // 2}")
    
    st.markdown("---")
    st.markdown("### Graph Structure")
    if st.button("📊 Export Graph as PNG", key="export_graph"):
        graph_path = export_graph_png("graph_structure.png")
        if graph_path:
            with open(graph_path, 'rb') as f:
                st.download_button(
                    label="📥 Download Graph PNG",
                    data=f.read(),
                    file_name="graph_structure.png",
                    mime="image/png",
                    key="download_graph"
                )

if st.session_state.messages:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message">{message["content"]}</div>', unsafe_allow_html=True)

if st.session_state.session_active:
    with st.form(key="query_form", clear_on_submit=True):
        query = st.text_input("Ask a question about compliance, risk, or internal policies...", key="query_input")
        submit_button = st.form_submit_button(label="Send")
    
    if submit_button and query:
        st.session_state.messages.append({"role": "user", "content": query})
        st.markdown(f'<div class="user-message">{query}</div>', unsafe_allow_html=True)
        
        with st.spinner("Processing your query..."):
            try:
                result = run_query(query)
                answer = result.get("answer", "I apologize, but I couldn't generate an answer.")
                log_trace(st.session_state.artifact_file, query, result)
                st.markdown(f'<div class="bot-message">{answer}</div>', unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                error_msg = f"I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        st.experimental_rerun()
else:
    st.warning("Session ended. Please restart to continue.")
    
    if "artifact_file" in st.session_state and st.session_state.artifact_file.exists() and st.session_state.get("artifact_ready", False):
        with open(st.session_state.artifact_file, 'rb') as f:
            artifact_bytes = f.read()
            st.download_button(
                label="📥 Download Artifact",
                data=artifact_bytes,
                file_name=st.session_state.artifact_file.name,
                mime="application/json",
                key="download_artifact_final"
            )


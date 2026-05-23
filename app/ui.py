import streamlit as st
import requests
from src.inference.utils import get_base64_of_bin_file
from configs.inference.settings import app_settings

front_end_settings = app_settings.front_end

# Configuration
st.set_page_config(
    page_title="Chefformer - A Recipe Generation Decoder Transformer",
    page_icon="👨‍🍳",
    layout="centered"
)
# Load Local Background 
img_base64 = get_base64_of_bin_file(front_end_settings.wallpaper_path)

# Custom Styling 
st.markdown(f"""
    <style>
    /* Target the main container for the background image */
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/png;base64,{img_base64}");
        background-size: cover;
        background-position: left center; /* Adjusted image to the left */
        background-attachment: fixed;
    }}

    /* Make the top header and bottom input container transparent 
    [data-testid="stHeader"], [data-testid="stBottom"] {{
        background-color: rgba(0,0,0,0) !important;
    }}
    */

    /* Style Title and Caption for high contrast */
    h1 {{
        color: white !important;
        text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.8);
        font-weight: 700;
    }}

    [data-testid="stCaptionContainer"] {{
        color: rgba(255, 255, 255, 0.9) !important;
        text-shadow: 1px 1px 4px rgba(0, 0, 0, 0.8);
    }}

    /* Add a slight blur or semi-transparent background to chat bubbles for readability */
    .stChatMessage {{
        background-color: rgba(255, 255, 255, 0.95) !important;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        color: #1E1E1E; /* Ensure text is dark enough on the light bubble */
    }}

    /* Make the error notification fully opaque and high-contrast */
    [data-testid="stAlert"] {{
        background-color: white !important;
        color: #B00020 !important; /* Professional dark red for errors */
        border: 1px solid #B00020;
    }}
    </style>
    """, unsafe_allow_html=True)

st.title("👨‍🍳 Chefformer")
st.caption("Prompt the model to generate a recipe!")

# Sidebar Generation Parameters
st.sidebar.header("Generation Settings")
temperature = st.sidebar.slider("Temperature", 0.0, 2.0, 0.1, 0.05)
top_p = st.sidebar.slider("Top P", 0.0, 1.0, 1.0, 0.05)
top_k = st.sidebar.slider("Top K", 0, 500, 100, 1)
max_tokens = st.sidebar.slider("Max Tokens", 1, 512, 512, 1)
stop_seq_raw = st.sidebar.text_input("Stop Sequences (comma separated)", "")
stop_sequences = [s.strip() for s in stop_seq_raw.split(",")] if stop_seq_raw else None

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "error" not in st.session_state:
    st.session_state.error = None

# Chat Input
if prompt := st.chat_input("What are we cooking today?..."):
    # Clear previous history to start fresh for every prompt
    st.session_state.error = None
    st.session_state.messages = [{"role": "user", "content": prompt}]

    # Show the current interaction immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Model is cooking (pun intended)..."):
            try:
                payload = {
                    "message": prompt,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "max_tokens": max_tokens,
                    "stop_sequences": stop_sequences
                }
                response = requests.post(front_end_settings.chat_url, json=payload)
                
                if response.status_code == 200:
                    result = response.json().get("response", "I'm sorry, I couldn't process that.")
                    # Save assistant response to state
                    st.session_state.messages.append({"role": "assistant", "content": result})
                else:
                    st.session_state.error = f"Error: API returned status code {response.status_code}"
            except requests.exceptions.ConnectionError:
                st.session_state.error = "Could not connect to the API. Is your FastAPI server running at http://127.0.0.1:8000?"
    
    # Force a rerun to clean the UI and display only the final result
    st.rerun()

# Display Error if exists
if st.session_state.error:
    st.error(st.session_state.error)

# Display Current Interaction
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

import streamlit as st
import os
import time
from backend import create_rag_chain

# --- 1. Page Configuration & Custom CSS ---
st.set_page_config(page_title="AI Document Analyzer", page_icon="🧠", layout="wide")

# Injecting Custom CSS to make it look professional
st.markdown("""
<style>
    /* Hide the default Streamlit top menu and footer for a clean look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Style the sidebar */
    [data-testid="stSidebar"] {
        background-color: #1E1E1E;
        padding-top: 2rem;
    }
    
    /* Make the title pop */
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: -webkit-linear-gradient(45deg, #4facfe, #00f2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.2rem;
        color: #A0AEC0;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. Header Section ---
st.markdown('<h1 class="main-title">🧠 AI Smart Document Analyzer</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Upload any PDF and instantly extract insights, summaries, and answers.</p>', unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 3. Enhanced Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135692.png", width=80) # Add a cool icon
    st.header("📂 Document Control")
    
    uploaded_file = st.file_uploader("Upload your PDF here", type="pdf")
    
    if uploaded_file is not None:
        temp_path = "./temp.pdf"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
            with st.spinner("⏳ Analyzing document & building neural pathways..."):
                try:
                    st.session_state.rag_chain = create_rag_chain(temp_path)
                    st.session_state.current_file = uploaded_file.name
                    st.session_state.messages = [] 
                    st.success("✅ Document processed successfully!")
                    st.balloons() # Fun animation when upload succeeds!
                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()
    
    # Add a "How to use" expander to make it user-friendly
    with st.expander("💡 How to use this app"):
        st.markdown("""
        1. Upload a PDF document using the button above.
        2. Wait for the success message.
        3. Ask questions in the chat box below!
        *Example: "Summarize the key points in 5 bullets."*
        """)

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- 4. Main Chat Interface ---

# Empty State / Welcome Message
if len(st.session_state.messages) == 0:
    st.info("👋 Welcome! Upload a PDF on the left to get started. I am ready to answer your questions.")

# Display chat history with Avatars
for message in st.session_state.messages:
    # Use different avatars for user and AI
    avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# --- 5. Chat Input & Streaming Effect ---
if prompt := st.chat_input("Message the AI..."):
    
    # 1. User Message
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. AI Response
    with st.chat_message("assistant", avatar="🤖"):
        if "rag_chain" not in st.session_state:
            st.warning("Please upload a PDF document first!")
            response = "Please upload a PDF document first!"
        else:
            with st.spinner("Thinking..."):
                try:
                    # Fetch the response
                    response_dict = st.session_state.rag_chain.invoke({"input": prompt})
                    response = response_dict["answer"]
                    
                    # --- SIMULATED STREAMING EFFECT ---
                    # This makes the text appear word-by-word instead of a huge block at once
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    # Split the response into words and simulate typing
                    for chunk in response.split(" "):
                        full_response += chunk + " "
                        time.sleep(0.03) # Adjust speed here
                        message_placeholder.markdown(full_response + "▌")
                    
                    # Final output without the cursor
                    message_placeholder.markdown(full_response)
                    
                except Exception as e:
                    response = f"An error occurred: {e}"
                    st.error(response)
    
    # Save to history
    st.session_state.messages.append({"role": "assistant", "content": response})
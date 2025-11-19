import streamlit as st
import json
from data_analyzer import SpotifyDataAnalyzer
from lm_chatbot import LMChatBot

#CONFIGURATION
DATA_FILE = "SpotifyFeatures.csv"
PAGE_TITLE = "Spotify AI Chatbot"
PAGE_ICON = "🎵"

# --- PAGE SETUP ---
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide"
)

#SESSION STATE INITIALIZATION
if "messages" not in st.session_state:
    st.session_state.messages = []
if "analyzer" not in st.session_state:
    st.session_state.analyzer = None
if "chatbot" not in st.session_state:
    st.session_state.chatbot = LMChatBot()


#HELPER FUNCTIONS

def initialize_data():
    """Initialize the data analyzer class."""
    try:
        st.session_state.analyzer = SpotifyDataAnalyzer(DATA_FILE)
        return True
    except FileNotFoundError:
        st.error(f"❌ File '{DATA_FILE}' not found. Please ensure it is in the same directory.")
        return False
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return False


def display_image(img_base64):
    """Render a base64 encoded image."""
    if img_base64:
        st.markdown(
            f'<img src="data:image/png;base64,{img_base64}" style="max-width: 100%; height: auto;">',
            unsafe_allow_html=True
        )


def handle_function_call(call_data):
    analyzer = st.session_state.analyzer
    function_name = call_data['name']
    args = call_data.get('args', {})

    # Mapping function names to analyzer methods
    if function_name == "get_summary":
        content = analyzer.get_summary()
        return {"content": content, "image": None}

    elif function_name == "analyze_popularity":
        content, img = analyzer.analyze_popularity()
        return {"content": content, "image": img}

    elif function_name == "show_correlations":
        content, img = analyzer.show_correlations()
        return {"content": content, "image": img}

    elif function_name == "analyze_feature":
        feature = args.get('feature_name', 'danceability')
        content, img = analyzer.analyze_feature(feature)
        return {"content": content, "image": img}

    elif function_name == "predict_song":
        content = analyzer.predict_song(args)
        return {"content": content, "image": None}

    else:
        return {"content": f"Tool '{function_name}' not recognized.", "image": None}


def execute_manual_analysis(func_name, feature_name=None):
    args = {"feature_name": feature_name} if feature_name else {}
    result = handle_function_call({"name": func_name, "args": args})

    # Add to UI history
    st.session_state.messages.append({
        "role": "assistant",
        "content": result['content'],
        "image": result['image'],
        "type": "analysis"
    })
    # Add to LLM history context to prevent errors
    st.session_state.chatbot.conversation_history.append(("assistant", result['content']))
    st.rerun()


# --- SIDEBAR UI ---
with st.sidebar:
    st.title(f"{PAGE_ICON} {PAGE_TITLE}")
    st.markdown("---")

    # 1. Connection Status
    st.subheader("LM Studio Connection")
    if st.button("Test Connection"):
        if st.session_state.chatbot.test_connection():
            st.success("Connected to LM Studio!")
        else:
            st.error("Cannot connect to LM Studio. Check if server is running.")

    st.markdown("---")

    # 2. Quick Analysis Buttons
    st.subheader("Quick Analysis")
    if st.session_state.analyzer:
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Summary"):
                execute_manual_analysis("get_summary")
            if st.button("Popularity"):
                execute_manual_analysis("analyze_popularity")

        with col2:
            if st.button("Correlations"):
                execute_manual_analysis("show_correlations")
            if st.button("Danceability"):
                execute_manual_analysis("analyze_feature", "danceability")
    else:
        st.warning("Data not loaded yet.")

    st.markdown("---")

    # 3. Reset Controls
    if st.button("🗑Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chatbot.conversation_history = []
        st.rerun()

    # 4. Instructions
    st.markdown("###How to use:")
    st.info("""
    1. Ensure **LM Studio** is running (Server ON).
    2. Model must support instructions (e.g., Llama 3 8B).
    3. Ask questions: "Show me correlations" or "Tell me a joke".
    """)

#MAIN CHAT INTERFACE

st.title("Spotify Data AI Assistant")
st.markdown("Chat with AI about your Spotify dataset.")

# 1. Load Data (if not loaded)
if st.session_state.analyzer is None:
    if initialize_data():
        welcome_msg = "Spotify data loaded! Ready to analyze."
        st.session_state.messages.append({
            "role": "assistant",
            "content": welcome_msg,
            "type": "text"
        })
    else:
        st.stop()  # Stop execution if data fails to load

# 2. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "image" in message and message["image"]:
            display_image(message["image"])

# 3. Handle User Input
if prompt := st.chat_input("Ask about Spotify data or anything..."):

    # A. Display User Message Immediately
    st.session_state.messages.append({"role": "user", "content": prompt, "type": "text"})
    with st.chat_message("user"):
        st.markdown(prompt)

    # B. Process AI Response
    with st.chat_message("assistant"):
        with st.spinner("AI loading process..."):
            llm_response = st.session_state.chatbot.send_message(prompt)

        # CASE 1: Function Call (Tool Use)
        if llm_response.get("type") == "function_call":
            func_name = llm_response['name']
            func_args = llm_response['args']

            # Execute Logic
            tool_output = handle_function_call(llm_response)
            content = tool_output['content']
            img = tool_output['image']

            # Display Output
            st.markdown(f"**🛠️ Tool Used:** `{func_name}`")
            st.markdown(content)
            display_image(img)

            # Save to UI History
            st.session_state.messages.append({
                "role": "assistant",
                "content": content,
                "image": img,
                "type": "analysis"
            })

            # Save to AI Context (Fix for 400 Error)
            st.session_state.chatbot.conversation_history.append(("assistant", content))

        # CASE 2: Text Response (Chat)
        elif llm_response.get("type") == "text":
            response = llm_response['content']
            st.markdown(response)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "type": "text"
            })

        # CASE 3: Error
        else:
            st.error("Error: No valid response received from AI.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Powered by LM Studio")
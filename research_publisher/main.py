import os
import streamlit as st

try:
    # Prefer a proper function if it's defined in research_publisher.py
    from research_publisher import run_research_publisher as _run_research_publisher  # type: ignore
    HAS_RUNNER = True
except Exception:
    _run_research_publisher = None
    HAS_RUNNER = False


st.set_page_config(page_title="📰 Research Publisher", layout="wide")

# Session state
if "linkup_api_key" not in st.session_state:
    st.session_state.linkup_api_key = ""
if "messages" not in st.session_state:
    st.session_state.messages = []


def reset_chat() -> None:
    st.session_state.messages = []


# Sidebar: Linkup Configuration
with st.sidebar:
    col1, col2 = st.columns([1, 3])
    with col1:
        st.write("")
        st.image("https://avatars.githubusercontent.com/u/175112039?s=200&v=4", width=65)
    with col2:
        st.header("Linkup Configuration")
        st.write("Deep Web Search")

    st.markdown("[Get your API key](https://app.linkup.so/sign-up)", unsafe_allow_html=True)

    linkup_api_key = st.text_input("Enter your Linkup API Key", type="password")
    if linkup_api_key:
        st.session_state.linkup_api_key = linkup_api_key
        os.environ["LINKUP_API_KEY"] = linkup_api_key
        st.success("API Key stored successfully!")


# Header
col1, col2 = st.columns([6, 1])
with col1:
    st.markdown("<h2 style='color: #0066cc;'>📰 Research Publisher</h2>", unsafe_allow_html=True)
    powered_by_html = """
    <div style='display: flex; align-items: center; gap: 10px; margin-top: 5px;'>
        <span style='font-size: 20px; color: #666;'>Powered by</span>
        <img src=\"https://cdn.prod.website-files.com/66cf2bfc3ed15b02da0ca770/66d07240057721394308addd_Logo%20(1).svg\" width=\"80\">
        <span style='font-size: 20px; color: #666;'>and</span>
        <img src=\"https://framerusercontent.com/images/wLLGrlJoyqYr9WvgZwzlw91A8U.png?scale-down-to=512\" width=\"100\">
    </div>
    """
    st.markdown(powered_by_html, unsafe_allow_html=True)
with col2:
    st.button("Clear ↺", on_click=reset_chat)

st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)

# Chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Input and processing
if prompt := st.chat_input("Enter your publishing/research query..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not st.session_state.linkup_api_key:
        response = "Please enter your Linkup API Key in the sidebar."
    else:
        with st.spinner("Running research publisher... this may take a moment..."):
            try:
                if HAS_RUNNER and _run_research_publisher is not None:
                    response = _run_research_publisher(prompt)
                else:
                    response = (
                        "run_research_publisher(query) is not defined in research_publisher.py.\n"
                        "Please add it so this UI can call your pipeline."
                    )
            except Exception as e:  # noqa: BLE001
                response = f"An error occurred: {e}"

    with st.chat_message("assistant"):
        st.markdown("### 🧠 Final Answer", unsafe_allow_html=True)
        st.markdown(response, unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": response})

def main():
    print("Hello from research-publisher!")


if __name__ == "__main__":
    main()

import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="FinSolve Chatbot", page_icon=":material/chat:")
st.title("FinSolve Internal Chatbot")

# Create all empty variables
if "auth" not in st.session_state:
    st.session_state.auth = None
if "role" not in st.session_state:
    st.session_state.role = None
if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.header("Login")
    if st.session_state.auth is None:
        username= st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Log in"):
            try:
                resp = requests.get(
                    f"{API_URL}/login", auth=(username, password), timeout=10
                )
            except requests.RequestException as exc:
                st.error(f"Error connecting to API: {exc}")
            else:
                if resp.status_code == 200:
                    st.session_state.auth = (username, password)
                    st.session_state.role = resp.json()["role"]
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please try again.")

    else:
        st.success(f"Logged in as {st.session_state.auth[0]} ({st.session_state.role})")
        if st.button("Log out"):
            st.session_state.auth = None
            st.session_state.role = None
            st.session_state.messages = []
            st.rerun()

if st.session_state.auth is None:
    st.info("Please log in to use the chatbot.")
    st.stop()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about FinSolve internal data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    f"{API_URL}/chat",
                    json={"message": prompt},
                    auth=st.session_state.auth,
                    timeout=120,
                )
            except requests.RequestException as exc:
                answer = f"Error connecting to API: {exc}"
            else:
                answer = (
                    resp.json()["answer"]
                    if resp.status_code == 200
                    else f"Error {resp.status_code}: {resp.text}"
                )
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
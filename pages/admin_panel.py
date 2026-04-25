import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

FAQ_PATH = "FAQ.txt"
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")

st.set_page_config(page_title="Admin Panel — Marg's Minimart", page_icon="⚙️")
st.title("⚙️ Admin Panel")
st.caption("Marg's Minimart Assistant — FAQ Manager")

# --- Password gate ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    pwd = st.text_input("Enter admin password", type="password")
    if st.button("Log in"):
        if pwd == ADMIN_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

# --- Authenticated: FAQ editor ---
st.success("Logged in as admin.")

if st.button("Log out"):
    st.session_state.authenticated = False
    st.rerun()

st.divider()
st.subheader("Edit FAQ.txt")
st.caption("Each Q&A pair will be embedded into the chatbot's knowledge base on save.")

# Load current content
with open(FAQ_PATH, "r", encoding="utf-8") as f:
    current_content = f.read()

edited = st.text_area(
    "FAQ content",
    value=current_content,
    height=500,
    help="Use a consistent format — e.g. Q: ... / A: ... on separate lines."
)

col1, col2 = st.columns([1, 3])

with col1:
    if st.button("💾 Save & Rebuild", type="primary"):
        with open(FAQ_PATH, "w", encoding="utf-8") as f:
            f.write(edited)

        try:
            from chatbot import reload_vectorstore
            reload_vectorstore()
            st.success("FAQ saved and knowledge base rebuilt!")
        except Exception as e:
            st.warning(f"FAQ saved, but vector store reload failed: {e}")
            st.info("The new FAQ will load the next time the app restarts.")

with col2:
    st.info("Changes take effect immediately for new conversations.")

st.divider()
st.subheader("Preview current entries")
st.code(edited, language="text")
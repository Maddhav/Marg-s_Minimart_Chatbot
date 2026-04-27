import streamlit as st
import os
from dotenv import load_dotenv
from database import get_all_conversations

load_dotenv()

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")

st.set_page_config(page_title="Chat History — Marg's Minimart", page_icon="📋")
st.title("📋 Chat History")
st.caption("All customer conversations")

# Password gate
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

if st.button("Log out"):
    st.session_state.authenticated = False
    st.rerun()

st.divider()

rows = get_all_conversations()

if not rows:
    st.info("No conversations yet.")
else:
    # Group by session
    sessions = {}
    for session_id, role, content, timestamp in rows:
        if session_id not in sessions:
            sessions[session_id] = []
        sessions[session_id].append((role, content, timestamp))

    for session_id, messages in sessions.items():
        first_time = messages[0][2]
        with st.expander(f"Conversation — {first_time} | ID: {session_id[:8]}"):
            for role, content, timestamp in messages:
                if role == "user":
                    st.markdown(f"**🧑 Customer** `{timestamp}`")
                else:
                    st.markdown(f"**🤖 Gary** `{timestamp}`")
                st.markdown(content)
                st.divider()
import streamlit as st
import os
from dotenv import load_dotenv
from database import get_all_conversations

load_dotenv()

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")

st.set_page_config(page_title="Chat History — Marg's Minimart", page_icon="📋")

# Password gate
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown("## 🔐 Admin Access")
    st.caption("This area is restricted to Marg's Minimart staff only.")
    st.divider()
    pwd = st.text_input("Password", type="password", placeholder="Enter admin password")
    if st.button("Log in", type="primary"):
        if pwd == ADMIN_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password. Please try again.")
    st.stop()

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("## 📋 Customer Conversations")
    st.caption("Every question customers have asked Gary — organized by session.")
with col2:
    if st.button("Log out"):
        st.session_state.authenticated = False
        st.rerun()

st.divider()

rows = get_all_conversations()

if not rows:
    st.info("No conversations yet. Once customers start chatting, their messages will appear here.")
else:
    sessions = {}
    for session_id, role, content, timestamp in rows:
        if session_id not in sessions:
            sessions[session_id] = []
        sessions[session_id].append((role, content, timestamp))

    st.markdown(f"**{len(sessions)} conversation(s) recorded**")
    st.write("")

    for session_id, messages in sessions.items():
        first_time = messages[0][2]
        last_time = messages[-1][2]
        total_messages = len(messages)
        customer_messages = [m for m in messages if m[0] == "user"]

        # Session card header
        with st.expander(f"📅  {first_time}   ·   {total_messages} messages   ·   Session {session_id[:8]}"):

            # Session summary strip
            st.markdown(
                f"""
                <div style="background:#f0f2f6; padding:12px 16px; border-radius:8px; margin-bottom:16px;">
                    <span style="margin-right:24px;">🕐 <b>Started:</b> {first_time}</span>
                    <span style="margin-right:24px;">🕐 <b>Ended:</b> {last_time}</span>
                    <span>💬 <b>Customer messages:</b> {len(customer_messages)}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Messages
            for role, content, timestamp in messages:
                if role == "user":
                    st.markdown(
                        f"""
                        <div style="margin: 8px 0;">
                            <span style="font-size:12px; color:#888;">🧑 Customer · {timestamp}</span>
                            <div style="background:#e8f0fe; padding:10px 14px; border-radius:8px; margin-top:4px;">
                                {content}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div style="margin: 8px 0;">
                            <span style="font-size:12px; color:#888;">⛽ Gary · {timestamp}</span>
                            <div style="background:#f6f6f6; padding:10px 14px; border-radius:8px; margin-top:4px;">
                                {content}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
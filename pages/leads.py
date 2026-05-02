import streamlit as st
import os
from dotenv import load_dotenv
from database import get_all_leads

load_dotenv()

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")

st.set_page_config(page_title="Leads — Marg's Minimart", page_icon="📞")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown("## 🔐 Admin Access")
    pwd = st.text_input("Password", type="password")
    if st.button("Log in", type="primary"):
        if pwd == ADMIN_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("## 📞 Customer Leads")
    st.caption("Customers who shared their contact info for deals and promotions.")
with col2:
    if st.button("Log out"):
        st.session_state.authenticated = False
        st.rerun()

st.divider()

leads = get_all_leads()

if not leads:
    st.info("No leads yet. Once customers share their contact info it will appear here.")
else:
    st.markdown(f"**{len(leads)} lead(s) collected**")
    st.write("")
    for name, contact, timestamp in leads:
        st.markdown(
            f"""
            <div style="background:#f0f2f6; padding:14px 18px; border-radius:10px; margin-bottom:10px;">
                <span style="font-size:16px;">👤 <b>{name}</b></span><br>
                <span style="color:#444;">📬 {contact}</span><br>
                <span style="font-size:12px; color:#888;">🕐 {timestamp}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
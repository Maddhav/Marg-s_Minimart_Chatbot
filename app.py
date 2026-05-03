import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from database import init_db, save_message, save_lead
from groq import Groq
import uuid
import os

load_dotenv()
init_db()

st.set_page_config(
    page_title="Marg's Minimart Assistant",
    page_icon="⛽",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600&family=Inter:wght@400;500&display=swap');

/* Hide Streamlit default elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display: none;}
[data-testid="collapsedControl"] {display: none;}

/* Page background */
.stApp {
    background-color: #1a1208;
    background-image: 
        radial-gradient(ellipse at top, #2d1f0e 0%, #1a1208 60%),
        url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.02'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
}

/* Main container */
.main .block-container {
    max-width: 780px;
    padding: 0 1rem 2rem 1rem;
    margin: 0 auto;
}

/* Header */
.margs-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem 1rem;
    border-bottom: 1px solid rgba(180, 100, 40, 0.3);
    margin-bottom: 1.5rem;
}

.margs-header h1 {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 2.4rem;
    color: #f0c070;
    margin: 0 0 0.3rem 0;
    letter-spacing: 0.02em;
    text-shadow: 0 2px 20px rgba(240, 150, 40, 0.3);
}

.margs-header p {
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem;
    color: #a08060;
    margin: 0;
    letter-spacing: 0.03em;
}

.margs-badge {
    display: inline-block;
    background: rgba(180, 80, 30, 0.25);
    border: 1px solid rgba(180, 80, 30, 0.4);
    color: #e09050;
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    padding: 3px 12px;
    border-radius: 20px;
    margin-bottom: 0.8rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* Language selector */
.stSelectbox label {
    color: #a08060 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8rem !important;
}

.stSelectbox > div > div {
    background-color: #2a1a08 !important;
    border: 1px solid rgba(180, 100, 40, 0.3) !important;
    border-radius: 8px !important;
    color: #d4a060 !important;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: transparent !important;
    padding: 0.3rem 0 !important;
}

/* Assistant bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) .stMarkdown {
    background: #2a1a0a !important;
    border: 1px solid rgba(180, 100, 40, 0.25) !important;
    border-radius: 0px 16px 16px 16px !important;
    padding: 12px 16px !important;
    color: #e8d5b0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
    max-width: 85% !important;
}

/* User bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stMarkdown {
    background: #8b2a0a !important;
    border: 1px solid rgba(200, 80, 30, 0.4) !important;
    border-radius: 16px 16px 0px 16px !important;
    padding: 12px 16px !important;
    color: #f5e8d5 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
    max-width: 85% !important;
    margin-left: auto !important;
}

/* Chat input */
[data-testid="stChatInput"] {
    background: #2a1a08 !important;
    border: 1px solid rgba(180, 100, 40, 0.4) !important;
    border-radius: 12px !important;
}

[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #e8d5b0 !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #6a5040 !important;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #e09050 !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #1a1208; }
::-webkit-scrollbar-thumb { background: #4a2a10; border-radius: 3px; }

/* Divider */
hr {
    border-color: rgba(180, 100, 40, 0.2) !important;
}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="margs-header">
    <div class="margs-badge">⛽ Dawson Creek, BC</div>
    <h1>Marg's Minimart</h1>
    <p>Ask me anything — I'm Gary, your 24/7 virtual assistant</p>
</div>
""", unsafe_allow_html=True)

# --- Language selector ---
language = st.selectbox(
    "🌐 Language / ਭਾਸ਼ਾ / Langue / Idioma",
    ["English", "Français", "ਪੰਜਾਬੀ", "Español", "हिन्दी"],
    index=0
)
st.session_state.selected_language = language

@st.cache_resource
def load_chain():
    loader = TextLoader("FAQ.txt")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False}
    )
    vectorstore = Chroma.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()
    prompt = ChatPromptTemplate.from_template("""
You are Gary, a friendly and helpful assistant for Marg's Minimart gas station in Dawson Creek, BC.
You are warm, conversational and professional.
Answer the customer's question based only on the information below.
Keep answers concise but friendly. Use natural conversational language.
If you don't know the answer say "Great question! I'm not sure about that one — give us a call and we'll help you out!"

IMPORTANT: Detect the language the customer is writing in and always respond in that same language.
If they write in French, respond in French.
If they write in Punjabi, respond in Punjabi.
If they write in Spanish, respond in Spanish.
If they write in English, respond in English.
If they write in Hindi, respond in Hindi.

Context: {context}

Question: {question}
""")
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile"
    )
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

chain = load_chain()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "customer_name" not in st.session_state:
    st.session_state.customer_name = None
if "lead_captured" not in st.session_state:
    st.session_state.lead_captured = False
if "awaiting_lead" not in st.session_state:
    st.session_state.awaiting_lead = False

# --- Name capture ---
if not st.session_state.customer_name:
    with st.chat_message("assistant", avatar="⛽"):
        st.markdown("Hey there! 👋 I'm **Gary**, your virtual assistant at Marg's Minimart. Before we get started, what's your name?")
    name_input = st.chat_input("Type your name to get started...")
    if name_input:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        check = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": """You are a name detector. You only reply with NAME or NOT_NAME. Nothing else.

Examples:
"John" → NAME
"Priya" → NAME
"Madhav" → NAME
"Hi there" → NOT_NAME
"How are you?" → NOT_NAME
"What time do you open?" → NOT_NAME
"Hey" → NOT_NAME
"I need help" → NOT_NAME
"Sarah Connor" → NAME
"My name is John" → NOT_NAME
"123" → NOT_NAME

Only reply NAME or NOT_NAME. Nothing else."""
                },
                {
                    "role": "user",
                    "content": f'Is this a person\'s name? "{name_input}"'
                }
            ],
            temperature=0
        )
        result = check.choices[0].message.content.strip().upper()
        if "NOT_NAME" in result:
            with st.chat_message("assistant", avatar="⛽"):
                st.markdown("Oops! I just need your name first 😊 Something like **John** or **Priya** works great!")
        else:
            st.session_state.customer_name = name_input.strip().title()
            st.session_state.session_id = f"{st.session_state.customer_name}_{st.session_state.session_id[:6]}"
            save_message(st.session_state.session_id, "assistant", "Hey there! I'm Gary. What's your name?")
            save_message(st.session_state.session_id, "user", name_input)
            st.rerun()
    st.stop()

# --- Greeting ---
if len(st.session_state.messages) == 0:
    greeting = f"Nice to meet you, **{st.session_state.customer_name}**! 😊 How can I help you today? Ask me about our fuel, store hours, deli, lottery tickets — anything!"
    st.session_state.messages.append({"role": "assistant", "content": greeting})
    save_message(st.session_state.session_id, "assistant", greeting)

# --- Display messages ---
for message in st.session_state.messages:
    avatar = "⛽" if message["role"] == "assistant" else "🧑"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# --- Chat input ---
if prompt := st.chat_input("Ask Gary anything about Marg's Minimart..."):

    if st.session_state.awaiting_lead and not st.session_state.lead_captured:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        check = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": """You detect if a message contains a valid email address or phone number.
Reply ONLY with CONTACT or NOT_CONTACT. Nothing else.

Examples:
"test@gmail.com" → CONTACT
"647-555-1234" → CONTACT
"my email is john@gmail.com" → CONTACT
"No thanks" → NOT_CONTACT
"I don't want to" → NOT_CONTACT
"hell no" → NOT_CONTACT
"maybe later" → NOT_CONTACT"""
                },
                {
                    "role": "user",
                    "content": f'Does this contain a contact? "{prompt}"'
                }
            ],
            temperature=0
        )
        result = check.choices[0].message.content.strip().upper()

        if "NOT_CONTACT" in result:
            positive_words = ["yes", "sure", "ok", "okay", "yeah", "yep", "definitely", "of course", "why not"]
            if any(word in prompt.lower() for word in positive_words):
                save_message(st.session_state.session_id, "user", prompt)
                ask_msg = "Great! 😊 Please share your email or phone number and we'll add you to our list!"
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.messages.append({"role": "assistant", "content": ask_msg})
                save_message(st.session_state.session_id, "assistant", ask_msg)
                st.rerun()
            else:
                st.session_state.awaiting_lead = False
                st.session_state.lead_captured = True
                save_message(st.session_state.session_id, "user", prompt)
                no_worries_msg = f"No worries at all, {st.session_state.customer_name}! 😊 Is there anything else I can help you with?"
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.messages.append({"role": "assistant", "content": no_worries_msg})
                save_message(st.session_state.session_id, "assistant", no_worries_msg)
                st.rerun()
        else:
            save_lead(st.session_state.session_id, st.session_state.customer_name, prompt)
            st.session_state.lead_captured = True
            st.session_state.awaiting_lead = False
            save_message(st.session_state.session_id, "user", prompt)
            thank_msg = f"Thank you, {st.session_state.customer_name}! 🙌 You'll be the first to hear about our deals. Is there anything else I can help you with?"
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": thank_msg})
            save_message(st.session_state.session_id, "assistant", thank_msg)
            st.rerun()

    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_message(st.session_state.session_id, "user", prompt)
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="⛽"):
            with st.spinner("Gary is thinking..."):
                response = chain.invoke(prompt)
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
        save_message(st.session_state.session_id, "assistant", response)

        user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
        if user_messages == 2 and not st.session_state.lead_captured:
            lead_msg = f"By the way {st.session_state.customer_name}, would you like to receive our weekly deals and promotions? Just drop your email or phone number! 🎉"
            st.session_state.messages.append({"role": "assistant", "content": lead_msg})
            save_message(st.session_state.session_id, "assistant", lead_msg)
            st.session_state.awaiting_lead = True
            st.rerun()
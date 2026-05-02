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

st.set_page_config(page_title="Gas Station Assistant", page_icon="⛽")
st.title("⛽ Gas Station Assistant")
st.caption("Ask me anything about our store, fuel, and services!")

language = st.selectbox(
    "🌐 Choose your language / ਆਪਣੀ ਭਾਸ਼ਾ ਚੁਣੋ / Choisissez votre langue / Elige tu idioma / हिन्दी",
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
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": False})
    vectorstore = Chroma.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()
    prompt = ChatPromptTemplate.from_template("""
You are Gary, a friendly and helpful assistant for Marg's Minimart gas station.
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
    with st.chat_message("assistant"):
        st.markdown("Hey there! 👋 My name is Gary and I am an AI Chatbot. Welcome to **Marg's Minimart**. Before we get started, could I get your name?")
    name_input = st.chat_input("Type your name...")
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
            with st.chat_message("assistant"):
                st.markdown("Oops! I just need your name first 😊 Something like **John** or **Priya** works!")
        else:
            st.session_state.customer_name = name_input.strip().title()
            st.session_state.session_id = f"{st.session_state.customer_name}_{st.session_state.session_id[:6]}"
            save_message(st.session_state.session_id, "assistant", "Hey there! 👋 Welcome to Marg's Minimart. Could I get your name?")
            save_message(st.session_state.session_id, "user", name_input)
            st.rerun()
    st.stop()

# --- Greeting on first load ---
if len(st.session_state.messages) == 0:
    greeting = f"Nice to meet you, {st.session_state.customer_name}! 😊 How can I help you today?"
    st.session_state.messages.append({"role": "assistant", "content": greeting})
    save_message(st.session_state.session_id, "assistant", greeting)

# --- Display chat history ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat input ---
if prompt := st.chat_input("Ask a question..."):
    
    # If awaiting lead contact info
    if st.session_state.awaiting_lead and not st.session_state.lead_captured:
        save_lead(st.session_state.session_id, st.session_state.customer_name, prompt)
        st.session_state.lead_captured = True
        st.session_state.awaiting_lead = False
        save_message(st.session_state.session_id, "user", prompt)
        thank_msg = f"Thank you {st.session_state.customer_name}! 🙌 You'll be the first to hear about our deals. Now, is there anything else I can help you with?"
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": thank_msg})
        save_message(st.session_state.session_id, "assistant", thank_msg)
        st.rerun()

    # Normal question to Gary
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_message(st.session_state.session_id, "user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chain.invoke(prompt)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    save_message(st.session_state.session_id, "assistant", response)

    # Ask for lead after 2nd customer message
    user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
    if user_messages == 2 and not st.session_state.lead_captured:
        lead_msg = f"By the way {st.session_state.customer_name}, would you like to receive our weekly deals and promotions? Just drop your email or phone number and we'll keep you in the loop! 🎉"
        st.session_state.messages.append({"role": "assistant", "content": lead_msg})
        save_message(st.session_state.session_id, "assistant", lead_msg)
        st.session_state.awaiting_lead = True
        st.rerun()
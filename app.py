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
from database import init_db, save_message  # ← new
import uuid                                  # ← new
import os

load_dotenv()
init_db()                                    # ← new

st.set_page_config(page_title="Gas Station Assistant", page_icon="⛽")
st.title("⛽ Gas Station Assistant")
st.caption("Ask me anything about our store, fuel, and services!")

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
If you don't know the answer say "Great question! I'm not sure about that one — 
give us a call and we'll help you out!"

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

# Give each browser session a unique ID
if "session_id" not in st.session_state:        # ← new
    st.session_state.session_id = str(uuid.uuid4())  # ← new

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_message(st.session_state.session_id, "user", prompt)  # ← new
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chain.invoke(prompt)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    save_message(st.session_state.session_id, "assistant", response)  # ← new
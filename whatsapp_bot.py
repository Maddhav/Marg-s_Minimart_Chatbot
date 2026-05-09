from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

# Build Gary once on startup
def build_chain():
    loader = TextLoader("FAQ.txt")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    from langchain_community.embeddings import FastEmbedEmbeddings
    embeddings = FastEmbedEmbeddings()
    vectorstore = Chroma.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()

    prompt = ChatPromptTemplate.from_template("""
    You are Gary, a friendly and helpful assistant for Marg's Minimart gas station.
    You are warm, conversational and professional.
    Answer the customer's question based only on the information below.
    Keep answers concise but friendly. Use natural conversational language.
    If you don't know the answer say "Great question! I'm not sure about that one — give us a call and we'll help you out!"

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

chain = build_chain()

@app.route("/whatsapp", methods=["POST"])
def whatsapp_reply():
    incoming_msg = request.form.get("Body", "").strip()
    response = MessagingResponse()
    msg = response.message()

    try:
        reply = chain.invoke(incoming_msg)
    except Exception as e:
        reply = "Sorry, I'm having trouble right now. Please call us directly!"

    msg.body(reply)
    return str(response)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
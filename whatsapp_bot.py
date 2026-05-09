from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_groq import ChatGroq
from langchain.embeddings.base import Embeddings

from dotenv import load_dotenv

from typing import List
import hashlib
import os

load_dotenv()

app = Flask(__name__)


class SimpleEmbeddings(Embeddings):
    """Very lightweight embeddings"""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def _embed(self, text: str) -> List[float]:
        hash_val = hashlib.md5(text.lower().encode()).hexdigest()
        return [
            int(hash_val[i:i+2], 16) / 255.0
            for i in range(0, 32, 2)
        ]


def build_chain():
    loader = TextLoader("FAQ.txt")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    embeddings = SimpleEmbeddings()

    vectorstore = Chroma.from_documents(
        chunks,
        embeddings
    )

    retriever = vectorstore.as_retriever()

    prompt = ChatPromptTemplate.from_template("""
You are Gary, a friendly and helpful assistant for Marg's Minimart gas station. 
You are warm, conversational and professional. 

Answer the customer's question based only on the information below. 
Keep answers concise but friendly. Use natural conversational language. 

If you don't know the answer say "Great question!
I'm not sure about that one — give us a call and we'll help you out!"

Context:
{context}

Question:
{question}
""")

    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile"
    )

    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
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
        print(e)

        reply = (
            "Sorry, I'm having trouble right now. "
            "Please contact the store directly."
        )

    msg.body(reply)

    return str(response)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
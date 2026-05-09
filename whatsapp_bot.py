from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_groq import ChatGroq

from dotenv import load_dotenv

import os

load_dotenv()

app = Flask(__name__)

# Lazy load chain
chain = None


def build_chain():

    with open("FAQ.txt", "r", encoding="utf-8") as file:
        faq_text = file.read()

    prompt = ChatPromptTemplate.from_template("""
You are Gary, a friendly and professional assistant for Marg's Minimart gas station.

Answer customer questions ONLY using the FAQ information below.

Keep responses short, conversational, and helpful.

If the answer is not available, say:
"Great question! Please contact the store directly and we'll help you."

FAQ Information:
{faq}

Customer Question:
{question}
""")

    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile"
    )

    chain = (
        {
            "faq": lambda x: faq_text,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


@app.route("/")
def home():
    return "Gary WhatsApp Bot is running!"


@app.route("/whatsapp", methods=["POST"])
def whatsapp_reply():

    global chain

    if chain is None:
        chain = build_chain()

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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
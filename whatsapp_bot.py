from flask import Flask, request, jsonify
from flask_cors import CORS
from twilio.twiml.messaging_response import MessagingResponse
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv()

app = Flask(__name__)
CORS(app)

chain = None

def build_chain():
    with open("FAQ.txt", "r", encoding="utf-8") as file:
        faq_text = file.read()

    prompt = ChatPromptTemplate.from_template("""
You are Gary, a friendly and professional assistant for Marg's Minimart gas station.
Answer customer questions ONLY using the FAQ information below.
Keep responses short, conversational, and helpful.
If the answer is not available, say: "Great question! Please contact the store directly and we'll help you."

IMPORTANT: Detect the language the customer is writing in and always respond in that same language.
If they write in French, respond in French.
If they write in Punjabi, respond in Punjabi.
If they write in Spanish, respond in Spanish.
If they write in Hindi, respond in Hindi.
If they write in English, respond in English.

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

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Gary is online!", "message": "Marg's Minimart AI Assistant"})

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
        reply = "Sorry, I'm having trouble right now. Please contact the store directly."
    msg.body(reply)
    return str(response)

@app.route("/chat", methods=["POST"])
def chat():
    global chain
    if chain is None:
        chain = build_chain()
    data = request.get_json()
    message = data.get("message", "")
    try:
        reply = chain.invoke(message)
        return jsonify({"response": reply})
    except Exception as e:
        print(e)
        return jsonify({"response": "I'm having trouble right now. Please call us directly!"})

@app.route("/validate-name", methods=["POST"])
def validate_name():
    data = request.get_json()
    name = data.get("name", "").strip()
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        check = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": """You are a name detector. Reply ONLY with NAME or NOT_NAME.

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
"123" → NOT_NAME

Only reply NAME or NOT_NAME. Nothing else."""
                },
                {
                    "role": "user",
                    "content": f'Is this a person\'s name? "{name}"'
                }
            ],
            temperature=0
        )
        result = check.choices[0].message.content.strip().upper()
        is_name = "NOT_NAME" not in result
        return jsonify({"is_name": is_name, "name": name.title()})
    except Exception as e:
        print(e)
        return jsonify({"is_name": True, "name": name.title()})

@app.route("/capture-lead", methods=["POST"])
def capture_lead():
    data = request.get_json()
    session_id = data.get("session_id", "")
    name = data.get("name", "")
    contact = data.get("contact", "")
    try:
        from database import save_lead
        save_lead(session_id, name, contact)
        return jsonify({"message": f"Thank you {name}! 🙌 You'll be the first to hear about our deals. Is there anything else I can help you with?"})
    except Exception as e:
        print(e)
        return jsonify({"message": f"Thank you {name}! 🙌 You'll be the first to hear about our deals!"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
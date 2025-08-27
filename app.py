from flask import Flask, render_template, request, jsonify
from flask import session
import uuid
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)
load_dotenv()
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret")
DB_URL = os.getenv("CHAT_HISTORY_DB_URL", "sqlite:///./chat_history.db")

def get_history(session_id: str):
    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string=DB_URL,
    )


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

embedding = download_embeddings()

index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    embedding=embedding,
    index_name=index_name
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GEMINI_API_KEY,
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

rag_with_memory = RunnableWithMessageHistory(
    rag_chain,
    get_history,                     # function that returns a ChatMessageHistory
    input_messages_key="input",      # name of the user input field
    history_messages_key="chat_history",  # must match MessagesPlaceholder key
    output_messages_key="answer",    # key where the model’s reply is found
)

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/get', methods=['GET', 'POST'])
def chatbot_response():
    msg = request.form['msg']

    # Ensure we have a stable session id
    if 'sid' not in session:
        session['sid'] = str(uuid.uuid4())
    sid = session['sid']

    # Invoke the memory-enabled chain
    result = rag_with_memory.invoke(
        {"input": msg},
        config={"configurable": {"session_id": sid}}
    )

    return str(result["answer"])

if __name__ == '__main__':
    app.run(host = "0.0.0.0", port = 8080, debug = True)

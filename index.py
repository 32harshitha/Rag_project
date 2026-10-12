import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    api_key=os.getenv("GOOGLE_API_KEY")
)

vector_db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vector_db.as_retriever(
    search_kwargs={"k": 3}
)

llm = GoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0
)

prompt = ChatPromptTemplate.from_template("""
You are an AI assistant.

Answer only using the provided context.

If the answer is unavailable, say:
"I couldn't find this information in the uploaded documents."

Context:
{context}

Question:
{question}

Answer:
""")


def ask_question(query):

    results = retriever.invoke(query)

    context = "\n\n".join(
        doc.page_content for doc in results
    )

    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    answer = llm.invoke(final_prompt)

    return answer
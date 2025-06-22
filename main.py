
import os
import time
import requests
import fitz  # PyMuPDF
import tiktoken
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from dotenv import load_dotenv
from trafilatura import extract, fetch_url
from openai import OpenAI
from supabase import create_client
from langdetect import detect
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import streamlit as st

# --- Settings ---
load_dotenv()
BASE_URL = os.getenv("BASE_URL", "https://www.11mind.com/lfk-demo")

# --- API Key Validation ---
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")
if not supabase_url or not supabase_key or not openai_key:
    st.error("Environment variables SUPABASE_URL, SUPABASE_API_KEY, and OPENAI_API_KEY are required.")
    st.stop()

# --- Clients Initialization ---
supabase = create_client(supabase_url, supabase_key)
openai_cli = OpenAI(api_key=openai_key)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.8)

# --- Text Extraction & Chunking ---
def extract_pdf_pages(data):
    """Extract text from each PDF page and return list of (page_number, text)."""
    try:
        doc = fitz.open(stream=data, filetype="pdf")
        return [(i+1, page.get_text().strip()) for i, page in enumerate(doc) if page.get_text().strip()]
    except Exception:
        return []


def extract_html_text(url):
    """Fetch HTML content and extract clean text."""
    raw = fetch_url(url)
    return extract(raw) if raw else ""


def chunk_text(text, max_tokens=365):
    """Split text into chunks of up to max_tokens tokens."""
    enc = tiktoken.get_encoding("cl100k_base")
    toks = enc.encode(text)
    return [enc.decode(toks[i:i+max_tokens]) for i in range(0, len(toks), max_tokens)]

# --- Crawler & Indexing ---
def find_links(base):
    """Return list of HTML and PDF links found on the base URL."""
    try:
        r = requests.get(base, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.content, "html.parser")
        htmls, pdfs = set(), set()
        for a in soup.find_all("a", href=True):
            full = urljoin(base, a["href"].strip())
            if full.lower().endswith(".pdf"): pdfs.add(full)
            elif full.startswith(base): htmls.add(full)
        return list(htmls), list(pdfs)
    except Exception:
        return [], []


def save_chunks(url, chunks, embeddings, page=None):
    """Save text chunks and their embeddings to Supabase table 'documents'."""
    rows = []
    for i, (txt, emb) in enumerate(zip(chunks, embeddings)):
        metadata = {"source_url": url, "page": page if page else i+1}
        rows.append({
            "content": txt,
            "metadata": metadata,
            "embedding": emb
        })
    supabase.table("documents").insert(rows).execute()


def process_url(url, idx, is_pdf=False):
    """Process each URL: extract content, chunk it, generate embeddings, save to Supabase."""
    try:
        if is_pdf:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            for pg, txt in extract_pdf_pages(r.content):
                chunks = chunk_text(txt)
                resp = openai_cli.embeddings.create(model="text-embedding-3-small", input=chunks)
                save_chunks(url, chunks, [d.embedding for d in resp.data], page=pg)
        else:
            txt = extract_html_text(url)
            if not txt:
                return
            chunks = chunk_text(txt)
            resp = openai_cli.embeddings.create(model="text-embedding-3-small", input=chunks)
            save_chunks(url, chunks, [d.embedding for d in resp.data])
    except Exception as e:
        st.warning(f"Processing error for {url}: {e}")

# --- Initialization of Data and Vector Store ---
@st.cache_resource(show_spinner=False)
def init_vectorstore_and_data():
    """Crawl site, index documents, and initialize SupabaseVectorStore."""
    htmls, pdfs = find_links(BASE_URL)
    for i, u in enumerate(htmls):
        process_url(u, i, False)
        time.sleep(1)
    for i, u in enumerate(pdfs):
        process_url(u, i, True)
        time.sleep(1)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = SupabaseVectorStore(
        client=supabase,
        embedding=embeddings,
        table_name="documents",
        query_name="match_documents"
    )
    return vs

# initialize vectorstore and load data
vectorstore = init_vectorstore_and_data()

# --- Question Answering Function ---
# Using ChatOpenAI model initialized above
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.8)

def ask_question(question):
    lang = detect(question)
    
    # İngilizce veya Almanca dışındaki diller için çift dilli uyarı
    if lang not in ["en", "de"]:
        answer = (
            "Please ask your question in English or German.\n"
            "Bitte stellen Sie Ihre Frage auf Englisch oder Deutsch."
        )
        return answer, []

    # Benzerlik araması yap
    docs = vectorstore.similarity_search(question, k=10)
    docs_has_content = bool(docs and any(doc.page_content.strip() for doc in docs))
    context = "\n\n".join(doc.page_content for doc in docs) if docs_has_content else ""

    # Almanca sorular için işlemler
    if lang == "de":
        if not docs_has_content:
            answer = (
                "Es tut mir leid, ich kann zu diesem Thema keine Auskunft geben, "
                "da es außerhalb unseres Inhaltsbereichs liegt. Bitte stellen Sie eine andere Frage."
            )
            return answer, []
        
        prompt = (
            f"Frage: {question}\n\n"
            f"Basierend auf dem folgenden Inhalt, gib eine detaillierte und klare Antwort:\n{context}\n\n"
            f"Bitte antworte in nummerierten, klaren Stichpunkten.\n"
            f"Wenn die Frage themenfremd oder außerhalb des Umfangs ist, antworte bitte höflich:\n"
            f"\"Es tut mir leid, ich kann zu diesem Thema keine Auskunft geben, da es außerhalb unseres Inhaltsbereichs liegt. "
            f"Bitte stellen Sie eine andere Frage.\"\n"
            f"In solchen Fällen bitte keine Quelleninformationen angeben."
        )
    
    # İngilizce sorular için işlemler
    else:
        if not docs_has_content:
            answer = (
                "Sorry, I cannot assist with this topic as it is outside the content scope. "
                "Feel free to ask about something else."
            )
            return answer, []
        
        prompt = (
            f"Question: {question}\n\n"
            f"Based on the following content, provide a detailed and clear answer:\n{context}\n\n"
            f"Please respond in clear and numbered bullet points.\n"
            f"If the question is unrelated or outside the scope, kindly respond with:\n"
            f"\"Sorry, I cannot assist with this topic as it is outside the content scope.\n"
            f"Feel free to ask about something else.\"\n"
            f"In such cases, please do not provide source information."
        )

    # Dil modelini çağır
    response = llm.invoke(prompt)
    answer = response.content.strip()

    # Gereksiz özür ifadelerini temizle (içerik olduğu halde)
    if docs_has_content:
        lower_answer = answer.lower()
        if (
            lower_answer.startswith("sorry") 
            or "outside the content scope" in lower_answer
            or "es tut mir leid" in lower_answer
        ):
            removal_phrases = [
                "sorry, i cannot assist with this topic as it is outside the content scope.",
                "feel free to ask about something else.",
                "es tut mir leid, ich kann zu diesem thema keine auskunft geben, da es außerhalb unseres inhaltsbereichs liegt.",
                "bitte stellen sie eine andere frage."
            ]
            for phrase in removal_phrases:
                lower_answer = lower_answer.replace(phrase, "")
            lower_answer = lower_answer.strip()
            
            # İlk harfi büyüt ve formatı düzelt
            answer = lower_answer.capitalize() if lower_answer else ""

    return answer, docs




# --- Streamlit UI ---
st.title("Streamlit RAG Chatbot")
question = st.text_input("Enter your question:")
if question:
    with st.spinner("Searching..."):
        ans, srcs = ask_question(question)
    st.subheader("Answer:")
    st.write(ans)
    
    # Eğer cevapta "sorry" veya "Es tut mir leid" gibi ifadeler varsa kaynakları gösterme
    lower_ans = ans.lower()
    if srcs and not (
        "sorry" in lower_ans or
        "es tut mir leid" in lower_ans or
        "cannot assist" in lower_ans or
        "keine auskunft" in lower_ans or
        "outside the content scope" in lower_ans
    ):
        st.subheader("Sources:")
        seen = set()
        for d in srcs:
            url = d.metadata.get("source_url")
            page = d.metadata.get("page")
            if (url, page) not in seen:
                seen.add((url, page))
                st.write(f"- {url} (Page: {page})")


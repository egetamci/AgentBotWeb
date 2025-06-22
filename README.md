# 🤖 LfK RAG Chatbot – AI-Assisted Outpatient Care Knowledge Assistant

This project delivers a **Retrieval-Augmented Generation (RAG) chatbot** designed specifically for assisting with content related to outpatient care regulations under **SGB V & XI** in Germany. It allows users to ask natural language questions (in **English or German**) about documents provided by [LfK Fördergesellschaft für ambulante Pflegedienste mbH](https://www.11mind.com/lfk-demo) and receive clear, source-based answers.

---

## 🚀 Features

- ✅ Supports **German and English** question answering  
- 📚 Context-aware responses based on vector similarity search over HTML and PDF content  
- 🧠 Powered by **OpenAI GPT (gpt-3.5-turbo)** and **text-embedding-3-small**  
- 🔎 Uses **Supabase** as a vector store for fast semantic search  
- 🕵️ Crawls both **HTML pages** and **PDF documents** for content ingestion  
- ❌ Gracefully handles irrelevant or unsupported questions with polite fallback messages  
- 🌐 Streamlit interface for fast deployment and local testing

---

## 🧠 Use Case

This chatbot is particularly suited for:

- Home care service providers
- Health professionals and consultants
- Regulatory compliance officers
- Users needing fast answers from long policy documents (50+ pages)

---

## 🛠️ Tech Stack

| Component | Description |
|----------|-------------|
| 🧠 OpenAI | GPT for response generation and embeddings |
| 📚 Supabase | Postgres + Vector Store backend |
| 📄 PyMuPDF | PDF content extraction |
| 🌍 Trafilatura | HTML text extraction |
| 🧵 Langchain | Vector search interface |
| 🎛️ Streamlit | UI and interaction layer |
| 🌐 Requests/BS4 | Web crawling and document collection |

---

## 📦 Installation

1. **Clone this repo:**
   ```bash
   git clone https://github.com/your-username/lfk-rag-chatbot.git
   cd lfk-rag-chatbot
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

3. **Install dependencies:**
   ```bash
   pip3 install -r requirements.txt
   ```

4. **Add your `.env` file:**
   Create a `.env` file in the project root with the following variables:

   ```env
   OPENAI_API_KEY=your_openai_key
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_API_KEY=your_supabase_service_role_key
   BASE_URL=https://www.11mind.com/lfk-demo
   ```

---

## 🧪 How to Run

```bash
streamlit run main.py
```

You’ll see a web interface where you can input questions in **English or German**. The bot will respond with answers derived from indexed documents and cite the sources—**only if relevant**.

---

## 🌍 Multilingual Support Logic

- If question is in **German**:
  - Prompt, answer, and UI labels are all in **German**
- If question is in **English**:
  - Everything stays in **English**
- If any other language:
  - Bot politely asks to switch to English or German

---

## 📎 Document Processing

The crawler handles:

- `.pdf` documents: Extracts and chunks per page
- `.html` content: Clean text scraping with `trafilatura`
- Embeddings are stored with metadata (`source_url`, `page`) in Supabase

---

## 🧼 Intelligent Filtering

- Questions with no relevant document context return polite fallback messages  
- If answer content is present, **"sorry" messages are filtered out**  
- **No source links shown** for out-of-scope or generic questions

---

## 🤝 Credits

Developed by **[Your Name]** in collaboration with LfK Fördergesellschaft.  
Inspired by real-world needs in digital transformation of outpatient care services.

---

## 🛡️ License

This project is licensed under the MIT License.  
Feel free to contribute, fork, or adapt to your organization.

---

## 🧩 Example Questions

- "Welche Leistungen der ambulanten Pflege nach § 36 SGB XI können wir anbieten?"
- "Can the chatbot assist with application processes under SGB XI?"
- "What kind of challenges do outpatient care providers face?"

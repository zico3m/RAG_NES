import os
from fastapi import FastAPI
from pydantic import BaseModel
from supabase import create_client
from groq import Groq

# Environment variables
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_ANON_KEY"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

# Clients
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
llm = Groq(api_key=GROQ_API_KEY)

app = FastAPI()

class AskRequest(BaseModel):
    question: str


def build_prompt(context, question):
    return f"""
أنت مساعد إخباري يعتمد فقط على الأخبار التالية.

قواعد الإجابة:
- لا تستخدم أي معرفة خارج النص.
- لا تفترض أو تحلل من عندك.
- إذا احتوت الأخبار على أي معلومات مرتبطة بالسؤال، لخصها حتى لو كانت جزئية.
- لا تقل إنه لا توجد معلومات طالما أن الأخبار تحتوي على محتوى مرتبط.
- ابدأ الإجابة دائمًا بالجملة التالية حرفيًا:
"تشير الأخبار إلى ما يلي:"

الأخبار:
{context}

السؤال:
{question}

الإجابة:
"""


def embed_query(text: str):
    """
    Embedding خفيف باستخدام Groq
    """
    emb = llm.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return emb.data[0].embedding


@app.post("/ask")
def ask_news(req: AskRequest):
    question = req.question

    # embedding للسؤال فقط (خفيف)
    query_embedding = embed_query("query: " + question)

    # البحث في Supabase (pgvector)
    results = supabase.rpc(
        "match_news_chunks",
        {
            "query_embedding": query_embedding,
            "match_count": 5
        }
    ).execute()

    context = ""
    for row in (results.data or []):
        context += row["chunk_text"][:800] + "\n\n---\n\n"

    prompt = build_prompt(context, question)

    response = llm.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return {
        "answer": response.choices[0].message.content
    }

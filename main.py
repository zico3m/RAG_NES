import os
from fastapi import FastAPI
from pydantic import BaseModel
from supabase import create_client
from sentence_transformers import SentenceTransformer
from groq import Groq

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_ANON_KEY"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
embed_model = SentenceTransformer("intfloat/multilingual-e5-small")
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

@app.post("/ask")
def ask_news(req: AskRequest):
    question = req.question

    query_embedding = embed_model.encode(
        ["query: " + question],
        normalize_embeddings=True
    )[0]

    # مهم: هذه RPC لازم تكون موجودة في Supabase
    results = supabase.rpc(
        "match_news_chunks",
        {
            "query_embedding": query_embedding.tolist(),
            "match_count": 5
        }
    ).execute()

    context = ""
    for row in (results.data or []):
        context += (row["chunk_text"][:800] + "\n\n---\n\n")

    prompt = build_prompt(context, question)

    response = llm.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return {"answer": response.choices[0].message.content}

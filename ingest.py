import os
from supabase import create_client
from sentence_transformers import SentenceTransformer

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_ANON_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
embed_model = SentenceTransformer("intfloat/multilingual-e5-small")

def chunk_text(text, max_words=200):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i+max_words]))
    return chunks

def ingest_news():
    response = (
        supabase
        .table("news")
        .select("id, title, content")
        .eq("status", "published")
        .eq("embedded", False)
        .execute()
    )

    news = response.data or []
    if not news:
        print("No new news.")
        return

    for idx, item in enumerate(news, start=1):
        title = item.get("title") or ""
        content = item.get("content") or ""
        full_text = title.strip() + "\n" + content.strip()

        chunks = chunk_text(full_text)
        if not chunks:
            # علّم الخبر حتى لا يعلق
            supabase.table("news").update({"embedded": True}).eq("id", item["id"]).execute()
            continue

        embeddings = embed_model.encode(chunks, normalize_embeddings=True)

        rows = []
        for ch, emb in zip(chunks, embeddings):
            rows.append({
                "news_id": item["id"],
                "chunk_text": ch,
                "embedding": emb.tolist()
            })

        supabase.table("news_chunks").insert(rows).execute()
        supabase.table("news").update({"embedded": True}).eq("id", item["id"]).execute()

        print(f"Processed {idx}/{len(news)} news (id={item['id']})")

if __name__ == "__main__":
    ingest_news()

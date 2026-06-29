from dotenv import load_dotenv
load_dotenv()
import chromadb
from openai import OpenAI
import json


client = OpenAI()
chroma = chromadb.EphemeralClient()

parts_col  = None
guides_col = None

def embed(text: str) -> list:
    return client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    ).data[0].embedding

def load_data():
    global parts_col, guides_col

    parts_col = chroma.get_or_create_collection("parts")
    with open("data/products.json") as f:
        parts = json.load(f)
    for p in parts:
        text = f"{p.get('name','')} {p.get('short_description','')} {' '.join(p.get('symptoms',[]))}"
        parts_col.add(
            ids=[str(p['part_number'])],
            embeddings=[embed(text)],
            documents=[text],
            metadatas=[{"part_number": str(p['part_number']),
                        "category": p.get('category','')}]
        )

    guides_col = chroma.get_or_create_collection("guides")
    with open("data/guides.json") as f:
        guides = json.load(f)
    for g in guides:
        guides_col.add(
            ids=[g['id']],
            embeddings=[embed(g['content'])],
            documents=[g['content']],
            metadatas=[{"type":  g.get('type',''),
                        "title": g.get('title',''),
                        "url":   g.get('url','')}]
        )

    print(f"[RAG] Indexed {len(parts)} parts, {len(guides)} guides")
    return len(parts)

def search_parts(query: str, n: int = 5) -> list:
    if not parts_col:
        return []
    r = parts_col.query(query_embeddings=[embed(query)], n_results=n)
    return r['documents'][0] if r['documents'] else []

def search_guides(query: str, guide_type: str = None, n: int = 3) -> list:
    if not guides_col:
        return []
    where = {"type": guide_type} if guide_type else None
    r = guides_col.query(
        query_embeddings=[embed(query)],
        n_results=n,
        where=where
    )
    return r['documents'][0] if r['documents'] else []
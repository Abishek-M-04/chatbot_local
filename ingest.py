print("📥 Ingesting medical documents...")

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np


# ----------------------------------
# CALLABLE LOCAL EMBEDDINGS (FIX)
# ----------------------------------
class LocalEmbeddings:
    def __call__(self, text):
        return self.embed_query(text)

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        np.random.seed(abs(hash(text)) % (2**32))
        return np.random.rand(384).tolist()


# ----------------------------------
# LOAD DATA
# ----------------------------------
with open("medical.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()


# ----------------------------------
# SPLIT TEXT
# ----------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_text(raw_text)


# ----------------------------------
# CREATE VECTORSTORE
# ----------------------------------
embeddings = LocalEmbeddings()

db = FAISS.from_texts(chunks, embeddings)

db.save_local("vectorstore")

print("✅ Vectorstore created successfully!")
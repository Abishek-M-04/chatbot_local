from langchain_community.vectorstores import FAISS
import numpy as np



# cAllablr EMBEDDINGS
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


embeddings = LocalEmbeddings()

db = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

print("🩺 Medical Chatbot (LOCAL RAG)")
print("Type 'exit' to quit\n")

while True:
    query = input("You: ")
    if query.lower() == "exit":
        break

    docs = db.similarity_search(query, k=2)

    print("\nBot:")
    for doc in docs:
        print("-", doc.page_content)

    print()

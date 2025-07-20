from sentence_transformers import SentenceTransformer
import pandas as pd
import os
import ijson
from torch import cuda
from datetime import datetime


def get_card_text(card):
    parts = [
        card.get("type_line", ""),
        ", ".join(card.get("keywords", [])),
        card.get("oracle_text", "")
    ]
    return "\n".join(p for p in parts if p.strip())


def create_embeddings(data_path, emb_path, model="all-MiniLM-L6-v2"):
    device = "cuda" if cuda.is_available() else "cpu"
    model = SentenceTransformer(model, device = device)
    embeddings = []
    texts = []
    with open(data_path, 'r', encoding='utf-8') as file:
        for card in ijson.items(file, 'item'):
            texts.append(get_card_text(card))
            embeddings.append({"id": card.get("id"),
                               "name": card.get("name"),
                               "color_id": card.get("color_identity"),
                               "colors": card.get("colors"),
                               "year": datetime.strptime(card.get("released_at"), "%Y-%m-%d").year, # Integer
                               "text": texts[-1],
                               "embedding":""})
    embedding_vectors = model.encode(texts, show_progress_bar=True)

    for i, vector in enumerate(embedding_vectors):
        embeddings[i]["embedding"] = embedding_vectors[i]

    df = pd.DataFrame(embeddings)
    df.to_pickle(emb_path)



if __name__ == "__main__":

    # # 1. Load a pretrained Sentence Transformer model
    # model = SentenceTransformer("all-MiniLM-L6-v2")

    # # The sentences to encode
    # sentences = [
    #     "The weather is lovely today.",
    #     "It's so sunny outside!",
    #     "He drove to the stadium.",
    # ]

    # # 2. Calculate embeddings by calling model.encode()
    # embeddings = model.encode(sentences)
    # print(embeddings.shape)
    # # [3, 384]

    # # 3. Calculate the embedding 
    # similarities = model.similarity(embeddings, embeddings)
    # print(similarities)
    # # tensor([[1.0000, 0.6660, 0.1046],similarities
    # #         [0.6660, 1.0000, 0.1411],
    # #         [0.1046, 0.1411, 1.0000]])

    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "monocol.json"))
    emb_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "all-MiniLM-L6-v2", "monocol_emb.pkl"))

    create_embeddings(data_path, emb_path)
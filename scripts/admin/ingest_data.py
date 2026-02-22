import os
import argparse
from datasets import load_dataset
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, Batch
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()


def ingest_agent_public_dataset(
    dataset_id: str, 
    collection_name: str, 
    embedding_col: str = None, 
    local_embed_model: str = None
):
    """
    Ingests a dataset into Qdrant. 
    Can use pre-embedded columns or calculate embeddings locally.
    """
    print(f"Loading dataset: {dataset_id}...")
    ds = load_dataset(dataset_id, split="train", streaming=True)

    client = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", 6333)),
    )

    embeddings_model = None
    if local_embed_model:
        print(f"Initializing local embeddings: {local_embed_model}")
        embeddings_model = HuggingFaceEmbeddings(model_name=local_embed_model)
        vector_size = 768 # Default for e5-base
    else:
        # Get a sample to determine vector size from pre-embedded column
        import json
        sample = next(iter(ds))
        emb_sample = sample[embedding_col]
        if isinstance(emb_sample, str):
            try:
                emb_sample = json.loads(emb_sample)
            except json.JSONDecodeError:
                emb_sample = [float(x.strip()) for x in emb_sample.strip("[]").split(",")]
        vector_size = len(emb_sample)

    if not client.collection_exists(collection_name):
        print(f"Creating collection: {collection_name} (Size: {vector_size})")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    print(f"Starting ingestion for {collection_name}...")
    batch_size = 50 if local_embed_model else 100
    ids, vectors, payloads = [], [], []

    import json
    count = 0
    for i, row in enumerate(ds):
        if embeddings_model:
            # Calculate embedding locally
            text = row.get("text") or row.get("content") or ""
            # e5 models often require 'query:' or 'passage:' prefix
            prefix = "passage: " if "e5" in local_embed_model else ""
            emb_value = embeddings_model.embed_query(prefix + text)
        else:
            emb_value = row[embedding_col]
            if isinstance(emb_value, str):
                try:
                    emb_value = json.loads(emb_value)
                except json.JSONDecodeError:
                    emb_value = [float(x.strip()) for x in emb_value.strip("[]").split(",")]

        ids.append(i)
        vectors.append(emb_value)
        payload = {k: v for k, v in row.items() if k != embedding_col}
        payloads.append(payload)

        if len(ids) >= batch_size:
            client.upsert(
                collection_name=collection_name,
                points=Batch(ids=ids, vectors=vectors, payloads=payloads),
            )
            count += len(ids)
            print(f"Ingested {count} points...")
            ids, vectors, payloads = [], [], []
            if count >= 2000: # Limit for local testing/demo purposes
                break

    if ids:
        client.upsert(
            collection_name=collection_name,
            points=Batch(ids=ids, vectors=vectors, payloads=payloads),
        )
        count += len(ids)

    print(f"Finished! Total {count} points ingested into {collection_name}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-embed", type=str, default=None, help="Local embedding model name")
    args = parser.parse_args()

    # Ingest Service Public
    ingest_agent_public_dataset(
        dataset_id="AgentPublic/service-public",
        collection_name="service_public_procedures",
        embedding_col="embeddings_bge-m3" if not args.local_embed else None,
        local_embed_model=args.local_embed
    )

    # Ingest LEGI
    ingest_agent_public_dataset(
        dataset_id="AgentPublic/legi",
        collection_name="legi_legislation",
        embedding_col="embeddings_bge-m3" if not args.local_embed else None,
        local_embed_model=args.local_embed
    )

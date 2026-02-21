import os
from datasets import load_dataset
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, Batch
from dotenv import load_dotenv

load_dotenv()


def ingest_agent_public_dataset(
    dataset_id: str, collection_name: str, embedding_col: str
):
    """
    Ingests a pre-embedded dataset from Hugging Face into Qdrant.
    """
    print(f"Loading dataset: {dataset_id}...")
    ds = load_dataset(dataset_id, split="train", streaming=True)

    client = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", 6333)),
    )

    # Get a sample to determine vector size
    import json

    sample = next(iter(ds))
    emb_sample = sample[embedding_col]

    # Correctly parse the sample to get the actual dimension (e.g., 1024)
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
    batch_size = 100
    ids = []
    vectors = []
    payloads = []

    import json

    count = 0
    for i, row in enumerate(ds):
        emb_value = row[embedding_col]
        # Parse if it's a string (Hugging Face sometimes stores these as JSON strings)
        if isinstance(emb_value, str):
            try:
                emb_value = json.loads(emb_value)
            except json.JSONDecodeError:
                # Fallback for simple string representations if json.loads fails
                emb_value = [float(x.strip()) for x in emb_value.strip("[]").split(",")]

        ids.append(i)
        vectors.append(emb_value)
        # Remove embedding from payload to save space
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

    # Final batch
    if ids:
        client.upsert(
            collection_name=collection_name,
            points=Batch(ids=ids, vectors=vectors, payloads=payloads),
        )
        count += len(ids)

    print(f"Finished! Total {count} points ingested into {collection_name}.")


if __name__ == "__main__":
    # Ingest Service Public (for procedures)
    ingest_agent_public_dataset(
        dataset_id="AgentPublic/service-public",
        collection_name="service_public_procedures",
        embedding_col="embeddings_bge-m3",
    )

    # Ingest LEGI (for law reference)
    ingest_agent_public_dataset(
        dataset_id="AgentPublic/legi",
        collection_name="legi_legislation",
        embedding_col="embeddings_bge-m3",
    )

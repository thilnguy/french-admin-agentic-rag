#!/usr/bin/env python3
import os
import argparse
import asyncio
from typing import List
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from src.config import settings
from src.utils.logger import logger
from langchain_huggingface import HuggingFaceEmbeddings

def parse_args():
    parser = argparse.ArgumentParser(description="Sync Legal Data from HuggingFace to Qdrant")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without writing to DB")
    parser.add_argument("--collection", default="service_public_procedures", help="Target Qdrant collection")
    return parser.parse_args()

def fetch_hf_data() -> List[Document]:
    logger.info(f"Loading dataset from HuggingFace: {settings.HF_DATASET_NAME}")
    try:
        # Assuming the dataset has a generic format or "train" split
        dataset = load_dataset(settings.HF_DATASET_NAME, split="train", token=settings.HUGGINGFACE_TOKEN)
        docs = []
        for row in dataset:
            # Adapt to the actual schema of the HuggingFace dataset
            text = row.get("text") or row.get("content", "")
            if not text:
                continue
            
            metadata = {k: v for k, v in row.items() if k not in ["text", "content"]}
            docs.append(Document(page_content=text, metadata=metadata))
            
        logger.info(f"Fetched {len(docs)} documents from HuggingFace.")
        return docs
    except Exception as e:
        logger.error(f"Failed to fetch dataset from HuggingFace: {e}")
        return []

async def sync_to_qdrant(docs: List[Document], collection_name: str, dry_run: bool):
    if not docs:
        logger.warning("No documents to sync.")
        return

    if dry_run:
        logger.info(f"[DRY RUN] Would sync {len(docs)} documents to collection '{collection_name}'.")
        return

    logger.info("Initializing Qdrant client and Embeddings model...")
    client = QdrantClient(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        api_key=settings.QDRANT_API_KEY,
        https=False
    )
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    # Ensure collection exists
    if not client.collection_exists(collection_name):
        logger.info(f"Creating collection '{collection_name}'...")
        # BAAI/bge-m3 has a dimension of 1024
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )

    store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
        content_payload_key="text",
    )

    logger.info(f"Upserting {len(docs)} documents into Qdrant...")
    # Add documents in batches to avoid overwhelming the DB
    batch_size = 100
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        await store.aadd_documents(batch)
        logger.info(f"  Synced {i+len(batch)}/{len(docs)}...")
        
    logger.info("Sync complete! Legal data is up to date.")

def main():
    args = parse_args()
    if not settings.HF_DATASET_NAME or settings.HF_DATASET_NAME == "your-hf-username/french-legal-data":
        logger.error("Please configure 'HF_DATASET_NAME' in your .env before running the sync.")
        return

    docs = fetch_hf_data()
    asyncio.run(sync_to_qdrant(docs, args.collection, args.dry_run))

if __name__ == "__main__":
    main()

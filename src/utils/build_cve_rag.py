"""
Build RAG Vector Database from CVE Data
Creates ChromaDB collection for semantic search of IoT vulnerabilities
"""

import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os
from pathlib import Path


def build_cve_vector_db():
    """Build vector database from CVE data"""

    print("🔨 Building CVE Vector Database...")

    # Paths
    base_dir = Path("/Users/nidhirajani/Desktop/DIC Phase 3")
    cve_file = base_dir / "data/external/iot_cves.csv"
    db_path = base_dir / "data/external/cve_vector_db"

    # Create directory
    db_path.mkdir(parents=True, exist_ok=True)

    # Load CVE data
    print("📊 Loading CVE data...")
    df = pd.read_csv(cve_file)
    print(f"   Loaded {len(df)} CVEs")

    # Initialize embedding model
    print("🤖 Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("   Model loaded")

    # Initialize ChromaDB
    print("💾 Initializing ChromaDB...")
    client = chromadb.PersistentClient(path=str(db_path))

    # Delete existing collection if it exists
    try:
        client.delete_collection("iot_cves")
        print("   Deleted existing collection")
    except:
        pass

    # Create new collection
    collection = client.create_collection(
        name="iot_cves",
        metadata={"description": "IoT vulnerability knowledge base"}
    )
    print("   Collection created")

    # Prepare documents
    print("📝 Preparing documents...")
    documents = []
    metadatas = []
    ids = []

    for idx, row in df.iterrows():
        # Create rich document text
        doc_text = f"""
CVE ID: {row['cve_id']}
CVSS Score: {row['cvss_score']}
Malware Family: {row['malware_family']}
Description: {row['description']}
"""
        documents.append(doc_text)

        # Metadata
        metadatas.append({
            'cve_id': row['cve_id'],
            'cvss_score': str(row['cvss_score']),
            'malware_family': row['malware_family'],
            'description': row['description'][:500]  # Truncate for metadata
        })

        ids.append(f"cve_{idx}")

    print(f"   Prepared {len(documents)} documents")

    # Generate embeddings
    print("🔢 Generating embeddings (this may take 1-2 minutes)...")
    embeddings = model.encode(documents, show_progress_bar=True)
    print("   Embeddings generated")

    # Add to collection in batches
    print("💿 Adding to ChromaDB...")
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        end_idx = min(i + batch_size, len(documents))
        collection.add(
            embeddings=embeddings[i:end_idx].tolist(),
            documents=documents[i:end_idx],
            metadatas=metadatas[i:end_idx],
            ids=ids[i:end_idx]
        )
        print(f"   Added batch {i // batch_size + 1}/{(len(documents) - 1) // batch_size + 1}")

    print(f"\n✅ Vector database built successfully!")
    print(f"   Location: {db_path}")
    print(f"   Total CVEs indexed: {collection.count()}")

    # Test query
    print("\n🧪 Testing semantic search...")
    test_query = "Mirai botnet telnet vulnerabilities"
    query_embedding = model.encode([test_query])
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=3
    )

    print(f"   Query: '{test_query}'")
    print(f"   Top 3 results:")
    for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        print(f"   {i + 1}. {meta['cve_id']} (CVSS: {meta['cvss_score']})")

    return db_path


if __name__ == "__main__":
    build_cve_vector_db()
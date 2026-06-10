import re
import pickle
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict

from tqdm import tqdm

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_postgres import PGVector
from sqlalchemy import create_engine, text

from embeddings import VertexGeminiEmbeddings
from config import (
    DATA_DIR,
    DOCUMENTS_PICKLE,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_LOCATION,
    GEMINI_PROJECT,
    PG_CONNECTION,
    PG_COLLECTION,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

EMBED_WORKERS = 24
INSERT_BATCH = 500


def parse_filename(filename: str) -> Dict:
    name_without_ext = filename.replace('.txt', '')

    course_match = re.match(r'([A-Z]{4})_(\d{4})_(\d{6})', name_without_ext)
    if course_match:
        dept = course_match.group(1)
        num = course_match.group(2)
        term = course_match.group(3)

        return {
            "type": "course_outline",
            "dept": dept,
            "course_num": num,
            "course_code": f"{dept} {num}",
            "term_code": term,
            "readable_name": f"{dept} {num}",
            "keywords": [dept.lower(), num, "course", "outline"]
        }

    if '_' in name_without_ext:
        tokens = name_without_ext.split('_')
        readable = ' '.join(tokens)
        return {
            "type": "program",
            "tokens": tokens,
            "readable_name": readable,
            "keywords": [t.lower() for t in tokens]
        }

    if '-' in name_without_ext:
        tokens = name_without_ext.split('-')
        readable = ' '.join(tokens)
        return {
            "type": "program",
            "tokens": tokens,
            "readable_name": readable,
            "keywords": [t.lower() for t in tokens]
        }

    return {
        "type": "general",
        "readable_name": name_without_ext,
        "keywords": [name_without_ext.lower()]
    }


def extract_enhanced_metadata(file_path: Path, content: str) -> Dict:
    relative_path = file_path.relative_to(DATA_DIR)
    category = relative_path.parts[0] if len(relative_path.parts) > 1 else "general"

    filename_info = parse_filename(file_path.name)

    metadata = {
        "source": str(file_path),
        "filename": file_path.name,
        "category": category,
        "file_size": len(content),
        "filename_keywords": ', '.join(filename_info["keywords"])
    }

    if filename_info["type"] == "course_outline":
        metadata["course_dept"] = filename_info["dept"]
        metadata["course_num"] = filename_info["course_num"]
        metadata["course_code"] = filename_info["course_code"]
        metadata["term_code"] = filename_info["term_code"]

    lines = content.split('\n')[:10]
    for line in lines:
        if line.startswith("URL:"):
            metadata["url"] = line.replace("URL:", "").strip()
        elif line.startswith("Title:"):
            metadata["title"] = line.replace("Title:", "").strip()
        elif line.startswith("Course:"):
            course_text = line.replace("Course:", "").strip()
            if not metadata.get("course_code"):
                metadata["course_code"] = course_text

    content_lower = content.lower()
    if "full-time" in content_lower or "full time" in content_lower:
        metadata["program_type"] = "full-time"
    elif "part-time" in content_lower or "flex" in content_lower:
        metadata["program_type"] = "part-time"
    else:
        metadata["program_type"] = "unknown"

    if "diploma" in content_lower:
        metadata["program_level"] = "diploma"
    elif "certificate" in content_lower:
        metadata["program_level"] = "certificate"
    elif "bachelor" in content_lower or "degree" in content_lower:
        metadata["program_level"] = "degree"

    level_match = re.search(r'(Level|Term)\s+(\d+)', content, re.IGNORECASE)
    if level_match:
        level_num = level_match.group(2)
        metadata["level"] = f"Level {level_num}"
        metadata["term"] = f"Term {level_num}"

    if "First Year" in content or "first year" in content:
        metadata["year"] = "First Year"
    elif "Second Year" in content or "second year" in content:
        metadata["year"] = "Second Year"

    credits_match = re.search(r'Credits?:\s*(\d+(?:\.\d+)?)', content)
    if credits_match:
        metadata["credits"] = float(credits_match.group(1))

    prereq_match = re.search(r'Prerequisites?:\s*([^\n]+)', content, re.IGNORECASE)
    if prereq_match:
        metadata["prerequisites"] = prereq_match.group(1).strip()

    return metadata


def create_content_prefix(filename: str, metadata: Dict) -> str:
    filename_info = parse_filename(filename)
    prefixes = []

    if filename_info["type"] == "course_outline":
        prefixes.append(f"[COURSE: {filename_info['readable_name']}]")
        prefixes.append(f"[TERM: {filename_info['term_code']}]")
    else:
        prefixes.append(f"[DOCUMENT: {filename_info['readable_name']}]")

    if metadata.get("program_type") and metadata["program_type"] != "unknown":
        prefixes.append(f"[{metadata['program_type'].upper()}]")

    if metadata.get("program_level"):
        prefixes.append(f"[{metadata['program_level'].upper()}]")

    if metadata.get("level"):
        level_num = metadata["level"].split()[-1]
        prefixes.append(f"[LEVEL {level_num}]")

    return ' '.join(prefixes)


def enhance_content(content: str, prefix: str, metadata: Dict) -> str:
    enhanced = f"{prefix}\n\n{content}"

    if metadata.get("level"):
        level_num = metadata["level"].split()[-1]
        enhanced = enhanced.replace(
            f"Level {level_num}",
            f"Level {level_num} (also known as Term {level_num}, Term-{level_num})",
            1
        )

    return enhanced


def load_documents() -> List[Document]:
    documents = []
    txt_files = list(DATA_DIR.rglob("*.txt"))

    print(f"Found {len(txt_files)} files")

    for file_path in tqdm(txt_files, desc="Processing", unit="file"):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Postgres TEXT cannot store NUL bytes
            content = content.replace('\x00', '')

            if not content.strip():
                continue

            metadata = extract_enhanced_metadata(file_path, content)
            prefix = create_content_prefix(file_path.name, metadata)
            enhanced_content = enhance_content(content, prefix, metadata)

            doc = Document(page_content=enhanced_content, metadata=metadata)
            documents.append(doc)

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

    print(f"Loaded {len(documents):,} documents")
    return documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    print(f"Chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )

    chunks = []
    for doc in tqdm(documents, desc="Chunking", unit="doc"):
        doc_chunks = text_splitter.split_documents([doc])
        chunks.extend(doc_chunks)

    sizes = [len(c.page_content) for c in chunks]
    print(f"Total chunks: {len(chunks):,}")
    print(f"Avg size: {sum(sizes) // len(sizes)} chars")
    print(f"Min: {min(sizes)} | Max: {max(sizes)}")

    return chunks


def save_documents(chunks: List[Document]):
    DOCUMENTS_PICKLE.parent.mkdir(parents=True, exist_ok=True)
    with open(DOCUMENTS_PICKLE, 'wb') as f:
        pickle.dump(chunks, f)
    print(f"Documents saved to {DOCUMENTS_PICKLE}")


def deterministic_ids(chunks: List[Document]) -> List[str]:
    # stable across re-runs so inserts upsert instead of duplicating
    counters = {}
    ids = []
    for chunk in chunks:
        source = chunk.metadata["source"]
        index = counters.get(source, 0)
        counters[source] = index + 1
        ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}#{index}")))
    return ids


def existing_ids(engine) -> set:
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT e.id FROM langchain_pg_embedding e
                    JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                    WHERE c.name = :name
                """),
                {"name": PG_COLLECTION}
            ).fetchall()
        return {row[0] for row in rows}
    except Exception:
        return set()


def embed_batch(embeddings: VertexGeminiEmbeddings, texts: List[str]) -> List[List[float]]:
    # gemini-embedding-001 takes 1 text per request: parallelize across threads,
    # each worker embeds a small slice sequentially (retries built into the client)
    slice_size = max(1, (len(texts) + EMBED_WORKERS - 1) // EMBED_WORKERS)
    slices = [texts[i:i + slice_size] for i in range(0, len(texts), slice_size)]
    results = [None] * len(slices)

    with ThreadPoolExecutor(max_workers=EMBED_WORKERS) as pool:
        futures = {
            pool.submit(embeddings.embed_documents, s): i
            for i, s in enumerate(slices)
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()

    return [vec for slice_vecs in results for vec in slice_vecs]


def build_vectorstore(chunks: List[Document]):
    embeddings = VertexGeminiEmbeddings(
        model_name=EMBEDDING_MODEL,
        project=GEMINI_PROJECT,
        location=EMBEDDING_LOCATION,
        dimensions=EMBEDDING_DIMENSIONS
    )

    engine = create_engine(PG_CONNECTION)
    store = PGVector(
        embeddings=embeddings,
        collection_name=PG_COLLECTION,
        connection=engine,
        use_jsonb=True,
        embedding_length=EMBEDDING_DIMENSIONS,
        create_extension=False
    )

    ids = deterministic_ids(chunks)

    done = existing_ids(engine)
    todo = [(i, c) for i, c in zip(ids, chunks) if i not in done]
    if done:
        print(f"Resuming: {len(done):,} chunks already in DB, {len(todo):,} remaining")

    progress = tqdm(total=len(todo), desc="Embedding + inserting", unit="chunk")
    for start in range(0, len(todo), INSERT_BATCH):
        batch = todo[start:start + INSERT_BATCH]
        batch_ids = [i for i, _ in batch]
        texts = [c.page_content for _, c in batch]
        metadatas = [c.metadata for _, c in batch]

        vectors = embed_batch(embeddings, texts)

        store.add_embeddings(
            texts=texts,
            embeddings=vectors,
            metadatas=metadatas,
            ids=batch_ids
        )
        progress.update(len(batch))
    progress.close()

    return engine


def create_hnsw_index(engine):
    print("Creating HNSW index (can take 10-40 min on a small instance)...")
    with engine.connect() as conn:
        conn = conn.execution_options(isolation_level="AUTOCOMMIT")
        conn.execute(text("SET maintenance_work_mem = '512MB'"))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_bcit_embedding_hnsw "
            "ON langchain_pg_embedding USING hnsw (embedding vector_cosine_ops)"
        ))
        conn.execute(text("ANALYZE langchain_pg_embedding"))
    print("HNSW index ready")


def main():
    if not DATA_DIR.exists():
        print(f"Error: Data directory not found: {DATA_DIR}")
        return

    documents = load_documents()
    chunks = chunk_documents(documents)

    save_documents(chunks)

    engine = build_vectorstore(chunks)
    create_hnsw_index(engine)

    print(f"Documents: {len(documents):,}")
    print(f"Chunks: {len(chunks):,}")


if __name__ == "__main__":
    main()

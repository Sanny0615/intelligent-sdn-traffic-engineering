"""
Document Ingestion & Chunking module.
Loads project markdown documents from docs/ and splits them into metadata-tagged text chunks.
"""

from dataclasses import dataclass
from typing import List
import os
import glob


@dataclass
class DocChunk:
    """Represents a text chunk extracted from project documentation."""
    chunk_id: str
    content: str
    source_file: str
    section_header: str


def load_and_chunk_docs(
    docs_dir: str = "docs",
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[DocChunk]:
    """
    Loads all markdown (.md) documents from docs_dir, parses header sections,
    and splits content into structured chunks with source metadata.
    """
    chunks: List[DocChunk] = []
    md_files = glob.glob(os.path.join(docs_dir, "*.md"))

    for filepath in md_files:
        filename = os.path.basename(filepath)
        rel_path = f"docs/{filename}"

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            continue

        # Split content into sections based on markdown headers (# or ##)
        lines = content.split("\n")
        current_header = "General Overview"
        current_buffer: List[str] = []

        for line in lines:
            if line.startswith("#"):
                # Save previous section if buffer has text
                if current_buffer:
                    section_text = "\n".join(current_buffer).strip()
                    if section_text:
                        _create_subchunks(
                            section_text, rel_path, current_header, chunks, chunk_size, chunk_overlap
                        )
                    current_buffer.clear()
                current_header = line.lstrip("#").strip()
            else:
                current_buffer.append(line)

        # Flush final buffer
        if current_buffer:
            section_text = "\n".join(current_buffer).strip()
            if section_text:
                _create_subchunks(
                    section_text, rel_path, current_header, chunks, chunk_size, chunk_overlap
                )

    return chunks


def _create_subchunks(
    text: str,
    source_file: str,
    header: str,
    chunk_list: List[DocChunk],
    chunk_size: int,
    chunk_overlap: int
) -> None:
    """Helper function to break large section text into overlapping chunks."""
    if len(text) <= chunk_size:
        cid = f"{source_file}#{header}#{len(chunk_list)}"
        chunk_list.append(DocChunk(
            chunk_id=cid,
            content=text,
            source_file=source_file,
            section_header=header
        ))
        return

    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        sub_text = text[start:end].strip()
        if sub_text:
            cid = f"{source_file}#{header}#{len(chunk_list)}"
            chunk_list.append(DocChunk(
                chunk_id=cid,
                content=sub_text,
                source_file=source_file,
                section_header=header
            ))
        start += (chunk_size - chunk_overlap)

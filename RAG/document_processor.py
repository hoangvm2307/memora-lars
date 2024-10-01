import os
import unicodedata
from typing import List
import fitz
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    UnstructuredImageLoader,
    UnstructuredHTMLLoader,
)


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("ASCII")


def load_documents(file_paths: List[str]) -> List[Document]:
    documents = []
    for file_path in file_paths:
        _, file_extension = os.path.splitext(file_path.lower())
        loader = get_loader(file_path, file_extension)
        if loader:
            documents.extend(loader.load())
        else:
            print(f"Unsupported file format: {file_extension}")
    return documents


def get_loader(file_path: str, file_extension: str):
    loaders = {
        ".pdf": PyPDFLoader,
        ".doc": Docx2txtLoader,
        ".docx": Docx2txtLoader,
        ".odt": Docx2txtLoader,
        ".rtf": UnstructuredWordDocumentLoader,
        ".txt": UnstructuredWordDocumentLoader,
        ".xls": UnstructuredExcelLoader,
        ".xlsx": UnstructuredExcelLoader,
        ".ods": UnstructuredExcelLoader,
        ".csv": UnstructuredExcelLoader,
        ".ppt": UnstructuredPowerPointLoader,
        ".pptx": UnstructuredPowerPointLoader,
        ".odp": UnstructuredPowerPointLoader,
        ".html": UnstructuredHTMLLoader,
    }
    image_extensions = [".bmp", ".gif", ".jpg", ".jpeg", ".png", ".svg", ".tiff"]

    if file_extension in loaders:
        return loaders[file_extension](file_path)
    elif file_extension in image_extensions:
        return UnstructuredImageLoader(file_path)
    return None


def extract_highlighted_text(
    pdf_path: str, page_num: int, start_char: int, end_char: int
) -> str:
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    start_rect = page.get_text("words")[start_char][:4]
    end_rect = page.get_text("words")[end_char - 1][:4]
    highlight_rect = fitz.Rect(start_rect[0], start_rect[1], end_rect[2], end_rect[3])
    highlighted_text = page.get_text("text", clip=highlight_rect)
    doc.close()
    return highlighted_text


def split_documents(
    documents: List[Document], chunk_size: int = 250, chunk_overlap: int = 20
) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_documents(documents)

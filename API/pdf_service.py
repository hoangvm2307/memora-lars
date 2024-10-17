from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_pdf(file):
    reader = PdfReader(file)
    pdf_texts = [p.extract_text().strip() for p in reader.pages]

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=500,   
        chunk_overlap=50,
        length_function=len
    )

    split_texts = text_splitter.split_text("\n\n".join(pdf_texts))

    return split_texts
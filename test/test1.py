from docling.document_converter import DocumentConverter
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

### 🔹 Docling PDF Parsing
def parse_with_docling(pdf_path):
    """
    Parses a PDF using Docling, extracts markdown content, 
    and prints the full extracted content.
    """
    try:
        # Ensure file exists
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File not found: {pdf_path}")

        # Initialize Docling Converter
        converter = DocumentConverter()
        markdown_document = converter.convert(pdf_path).document.export_to_markdown()

        # Define headers to split on (modify as needed)
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        
        # Initialize Markdown Splitter
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        docs_list = markdown_splitter.split_text(markdown_document)

        # Print full extracted sections
        print("\n✅ Full Extracted Content (Docling):")
        for idx, doc in enumerate(docs_list):
            print(f"\n🔹 Section {idx + 1}:\n{doc}\n" + "-"*80)

        return docs_list

    except Exception as e:
        print(f"\n❌ Error during Docling processing: {e}")
        return []

### 🔹 LangChain PDF Parsing
def parse_with_langchain(pdf_path):
    """
    Parses a PDF using LangChain's PyPDFLoader and prints the full extracted text.
    """
    try:
        # Ensure file exists
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File not found: {pdf_path}")

        # Load PDF using PyPDFLoader
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        # Extract text from all pages
        text = "\n\n".join([page.page_content for page in pages])

        # Print full extracted content
        print("\n✅ Full Extracted Content (LangChain):\n")
        print(text)
        print("\n" + "="*100)

        return text

    except Exception as e:
        print(f"\n❌ Error during LangChain processing: {e}")
        return ""

### 🔹 Main Execution
def main():
    ocr_path = "test/ocr_test.pdf"
    scanned_pdf_path = "test/sample.png"
    
    print("\n🔍 Running Docling Extraction for OCR...")
    docling_docs = parse_with_docling(ocr_path)

    print("\n🔍 Running LangChain Extraction for OCR...")
    langchain_text = parse_with_langchain(ocr_path)

    print("\n🔍 Running Docling Extraction for scanned PDF...")
    docling_docs = parse_with_docling(scanned_pdf_path)

    print("\n🔍 Running LangChain Extraction for scanned PDF...")
    langchain_text = parse_with_langchain(scanned_pdf_path)

if __name__ == "__main__":
    main()

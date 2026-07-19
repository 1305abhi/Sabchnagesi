import os
from pypdf import PdfReader

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text content from a PDF file.
    
    Args:
        pdf_path (str): The absolute or relative path to the PDF file.
        
    Returns:
        str: Extracted and cleaned text from the PDF.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a valid PDF or has no extractable text.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The file '{pdf_path}' does not exist.")
        
    if not pdf_path.lower().endswith('.pdf'):
        raise ValueError(f"The file '{pdf_path}' is not a PDF file.")
        
    try:
        reader = PdfReader(pdf_path)
    except Exception as e:
        raise ValueError(f"Failed to parse PDF file. It might be corrupted or encrypted: {e}")
        
    full_text = []
    
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            full_text.append(text)
            
    extracted = "\n".join(full_text).strip()
    
    if not extracted:
        raise ValueError(
            f"No text could be extracted from '{pdf_path}'. "
            "It might contain only scanned images/photos of text. "
            "SabChnagaSi requires digital text PDFs to perform analysis."
        )
        
    return extracted

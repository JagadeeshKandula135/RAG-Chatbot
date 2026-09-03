import re
import json
from pathlib import Path
from PyPDF2 import PdfReader


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(str(pdf_path))
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text

def extract_parent_title(full_text, parent_number):

    parent_title_match = re.search(
        rf"^{parent_number}\s+(.+)", 
        full_text, 
        re.MULTILINE
    )
    return parent_title_match.group(1).strip() if parent_title_match else ""


def parse_sections(text):
    heading_pattern = re.compile(r"^(\d+(?:\.\d+)*)\s+([A-Za-z].+)", re.MULTILINE)
    sections = []
    
    for match in heading_pattern.finditer(text):
        number = match.group(1)      # e.g., "3.2.1"
        title = match.group(2).strip()
        start_index = match.start()
        
        if "." in number:
            
            parent = number.split(".")[0]
            parent_title = extract_parent_title(text, parent)
            sections.append({
                "section": parent_title,
                "subsection": f"{number} {title}",
                "start": start_index
            })
        else:
            
            sections.append({
                "section": title,
                "start": start_index
            })
    
    return sections   

def find_abstract(text):
    abstract_match = re.search(r"\bAbstract\b", text)
    if abstract_match:
        return {"section": "Abstract", "start": abstract_match.start()}
    return None 

def extract_pdf_sections(full_text):
    
    sections = parse_sections(full_text)
    
    
    abstract = find_abstract(full_text)
    if abstract:
        sections.insert(0, abstract)
    
    return sections
    
    
    
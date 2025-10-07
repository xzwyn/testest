# document_aligner/src/processing/json_parser.py

import json
from pathlib import Path
from typing import List, Dict, Any

import config

# A type alias for our structured content for clarity
ContentItem = Dict[str, Any]

def _convert_table_to_markdown(table_obj: Dict) -> str:
    """Converts an Azure table object into a Markdown string."""
    markdown_str = ""
    if not table_obj.get('cells'):
        return ""

    # Create header
    header_cells = [cell for cell in table_obj['cells'] if cell.get('kind') == 'columnHeader']
    if header_cells:
        header_cells.sort(key=lambda x: x['columnIndex'])
        # Handle cells that might span multiple columns
        header_content = []
        for cell in header_cells:
            content = cell.get('content', '').strip()
            col_span = cell.get('columnSpan', 1)
            header_content.extend([content] * col_span)
        
        header_row = "| " + " | ".join(header_content) + " |"
        separator_row = "| " + " | ".join(["---"] * len(header_content)) + " |"
        markdown_str += header_row + "\n" + separator_row + "\n"

    # Create body rows
    body_cells = [cell for cell in table_obj['cells'] if cell.get('kind') is None]
    
    rows = {}
    for cell in body_cells:
        row_idx = cell.get('rowIndex', 0)
        if row_idx not in rows:
            rows[row_idx] = []
        rows[row_idx].append(cell)

    for row_idx in sorted(rows.keys()):
        row_cells = sorted(rows[row_idx], key=lambda x: x.get('columnIndex', 0))
        row_str = "| " + " | ".join([cell.get('content', '').strip() for cell in row_cells]) + " |"
        markdown_str += row_str + "\n"
        
    return markdown_str.strip()


def process_document_json(filepath: Path) -> List[ContentItem]:
    """
    Reads and processes an Azure Document Intelligence JSON file,
    now with dedicated handling for tables.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"The file '{filepath}' was not found.")

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    try:
        analyze_result = data['analyzeResult']
        full_text_content = analyze_result['content']
        raw_paragraphs = analyze_result.get('paragraphs', [])
        pages = analyze_result.get('pages', [])
        raw_tables = analyze_result.get('tables', [])
    except KeyError as e:
        raise ValueError(f"JSON file '{filepath}' is missing expected key: {e}") from e
    
    # --- Step 1: Identify all character offsets belonging to tables to avoid duplication ---
    table_offsets = set()
    for table in raw_tables:
        for span in table.get('spans', []):
            for i in range(span['offset'], span['offset'] + span['length']):
                table_offsets.add(i)
    
    # Identify all character offsets that are handwritten
    handwritten_offsets = set()
    if 'styles' in analyze_result:
        for style in analyze_result['styles']:
            if style.get('isHandwritten') and style.get('spans'):
                for span in style['spans']:
                    for i in range(span['offset'], span['offset'] + span['length']):
                        handwritten_offsets.add(i)
    
    # Create a quick lookup for page number by span offset
    page_lookup = {}
    for page in pages:
        for span in page.get('spans', []):
            for i in range(span['offset'], span['offset'] + span['length']):
                page_lookup[i] = page.get('pageNumber', 0)

    # --- Step 2: Extract all content, including tables, and sort by position ---
    all_content: List[ContentItem] = []

    # Process PARAGRAPHS
    for p in raw_paragraphs:
        role = p.get('role', 'paragraph')
        if role in config.IGNORED_ROLES or not p.get('spans'):
            continue
        
        offset = p['spans'][0]['offset']
        # If the paragraph is inside a table or is handwritten, SKIP it.
        if offset in table_offsets or offset in handwritten_offsets:
            continue
            
        length = p['spans'][0]['length']
        text = full_text_content[offset : offset + length].strip()
        page_number = page_lookup.get(offset, 0)
        if text:
            all_content.append({'text': text, 'type': role, 'page': page_number, 'offset': offset})
            
    # Process TABLES
    for table in raw_tables:
        if not table.get('spans'):
            continue
        offset = table['spans'][0]['offset']
        page_number = page_lookup.get(offset, 0)
        markdown_table = _convert_table_to_markdown(table)
        if markdown_table:
            all_content.append({'text': markdown_table, 'type': 'table', 'page': page_number, 'offset': offset})

    # Sort all extracted content by its character offset to maintain document order
    all_content.sort(key=lambda x: x['offset'])

    # --- Step 3: Stitch broken paragraphs ---
    final_content: List[ContentItem] = []
    stitched_text = ""
    current_page = 0
    current_type = "paragraph"

    for i, segment in enumerate(all_content):
        # If the current element is a table or a structural heading, finalize the previous stitched text.
        is_standalone = segment['type'] in config.STRUCTURAL_ROLES or segment['type'] == 'table'

        if is_standalone:
            if stitched_text: # Finalize any pending paragraph
                final_content.append({'text': stitched_text, 'type': current_type, 'page': current_page})
                stitched_text = ""
            final_content.append(segment) # Add the standalone item
            continue

        # This logic handles stitching of regular paragraphs
        if not stitched_text: # Start a new paragraph
            stitched_text = segment['text']
            current_page = segment['page']
            current_type = segment['type']
        else:
            # If previous text ends with punctuation, start a new paragraph
            if stitched_text.endswith(('.', '!', '?', ':', '•')):
                final_content.append({'text': stitched_text, 'type': current_type, 'page': current_page})
                stitched_text = segment['text']
                current_page = segment['page']
                current_type = segment['type']
            else: # Continue stitching the current paragraph
                stitched_text += f" {segment['text']}"

    # Add the last stitched paragraph if it exists
    if stitched_text:
        final_content.append({'text': stitched_text, 'type': current_type, 'page': current_page})
        
    return final_content
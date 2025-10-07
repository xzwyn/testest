from typing import List, Dict, Any
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm

import config
from src.reporting.excel_writer import save_calculation_report

ContentItem = Dict[str, Any]
AlignedPair = Dict[str, Any]

_model = None

def _get_model(model_name: str) -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Loading sentence transformer model '{model_name}'...")
        _model = SentenceTransformer(model_name)
    return _model

def _calculate_type_matrix(eng_content: List[ContentItem], ger_content: List[ContentItem]) -> np.ndarray:
    num_eng = len(eng_content)
    num_ger = len(ger_content)
    type_matrix = np.zeros((num_eng, num_ger))

    for i in range(num_eng):
        for j in range(num_ger):
            if eng_content[i]['type'] == ger_content[j]['type']:
                type_matrix[i, j] = config.TYPE_MATCH_BONUS
            else:
                type_matrix[i, j] = config.TYPE_MISMATCH_PENALTY
    return type_matrix

def _calculate_proximity_matrix(num_eng: int, num_ger: int) -> np.ndarray:
    proximity_matrix = np.zeros((num_eng, num_ger))
    for i in range(num_eng):
        for j in range(num_ger):
            norm_pos_eng = i / num_eng
            norm_pos_ger = j / num_ger
            proximity_matrix[i, j] = 1.0 - abs(norm_pos_eng - norm_pos_ger)
    return proximity_matrix

def align_content(
    english_content: List[ContentItem],
    german_content: List[ContentItem],
    generate_debug_report: bool = False,
    debug_report_path: Path = None
) -> List[AlignedPair]:
    if not english_content or not german_content:
        return []

    model = _get_model(config.MODEL_NAME)
    num_eng, num_ger = len(english_content), len(german_content)

    eng_texts = [item['text'] for item in english_content]
    ger_texts = [item['text'] for item in german_content]
    
    print("Generating embeddings...")
    english_embeddings = model.encode(eng_texts, convert_to_numpy=True, show_progress_bar=True)
    german_embeddings = model.encode(ger_texts, convert_to_numpy=True, show_progress_bar=True)
    
    print("Calculating score matrices (semantic, type, proximity)...")
    semantic_matrix = cosine_similarity(english_embeddings, german_embeddings)
    type_matrix = _calculate_type_matrix(english_content, german_content)
    proximity_matrix = _calculate_proximity_matrix(num_eng, num_ger)

    blended_matrix = (
        (config.W_SEMANTIC * semantic_matrix) +
        (config.W_TYPE * type_matrix) +
        (config.W_PROXIMITY * proximity_matrix)
    )

    if generate_debug_report and debug_report_path:
        print("Generating detailed calculation report for debugging...")
        save_calculation_report(
            english_content=english_content,
            german_content=german_content,
            blended_matrix=blended_matrix,
            semantic_matrix=semantic_matrix,
            type_matrix=type_matrix,
            proximity_matrix=proximity_matrix,
            filepath=debug_report_path
        )

    print("Finding best matches based on blended scores...")
    aligned_pairs: List[AlignedPair] = []
    used_german_indices = set()
    
    best_ger_matches = np.argmax(blended_matrix, axis=1)
    best_eng_matches = np.argmax(blended_matrix, axis=0)

    for eng_idx, ger_idx in enumerate(best_ger_matches):
        is_mutual_best_match = (best_eng_matches[ger_idx] == eng_idx)
        score = blended_matrix[eng_idx, ger_idx]

        if is_mutual_best_match and score >= config.SIMILARITY_THRESHOLD:
            semantic_score = semantic_matrix[eng_idx, ger_idx]
            aligned_pairs.append({
                "english": english_content[eng_idx],
                "german": german_content[ger_idx],
                "similarity": semantic_score
            })
            used_german_indices.add(ger_idx)

    matched_english_ids = {id(pair['english']) for pair in aligned_pairs if pair.get('english')}
    for item in english_content:
        if id(item) not in matched_english_ids:
            aligned_pairs.append({"english": item, "german": None, "similarity": 0.0})

    for idx, item in enumerate(german_content):
        if idx not in used_german_indices:
             aligned_pairs.append({"english": None, "german": item, "similarity": 0.0})

    aligned_pairs.sort(key=lambda x: x['english']['page'] if x.get('english') else float('inf'))
    
    return aligned_pairs
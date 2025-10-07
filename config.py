# document_aligner/config.py

# --- Model Configuration ---
# The name of the multilingual sentence transformer model to use for semantic similarity.
MODEL_NAME: str = 'distiluse-base-multilingual-cased-v2'

# --- Alignment Configuration ---
# The minimum BLENDED score required to consider two segments a match.
SIMILARITY_THRESHOLD: float = 0.70

# --- HYBRID SCORE WEIGHTS (Total should ideally sum to 1.0) ---
W_SEMANTIC: float = 0.7  # Weight of the semantic similarity score
W_TYPE: float = 0.2      # Weight of the structural type match score
W_PROXIMITY: float = 0.1   # Weight of the sequential proximity score

# --- TYPE SCORE CONFIGURATION ---
TYPE_MATCH_BONUS: float = 1.0   # Score given when types match (e.g., heading -> heading)
TYPE_MISMATCH_PENALTY: float = -1.0 # Score given when types mismatch (e.g., heading -> paragraph)

# --- Evaluation Configuration ---
# Default Azure OpenAI chat deployment name. This is overridden by the
# AZURE_OPENAI_DEPLOYMENT environment variable if it is set.
AZURE_CHAT_DEPLOYMENT: str = "gpt-4o"

# --- JSON Processing Configuration ---
IGNORED_ROLES = {"pageHeader", "pageFooter", "pageNumber"}
STRUCTURAL_ROLES = {'title', 'sectionHeading', 'subheading'}

# --- Directory Configuration ---
INPUT_DIR: str = "input"
OUTPUT_DIR: str = "output"
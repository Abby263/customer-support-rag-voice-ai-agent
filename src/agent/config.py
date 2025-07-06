import yaml
from pathlib import Path

# Define project root to resolve absolute paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

def load_config():
    """Loads the application configuration from the YAML file."""
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    return config

# Load the configuration once when the module is imported
config = load_config()

# Expose configuration sections as easily accessible variables
AGENT_CONFIG = config.get("agent", {})
PATHS_CONFIG = config.get("paths", {})
TEXT_SPLITTER_CONFIG = config.get("text_splitter", {})

# Resolve and export specific paths for convenience
DOCUMENTS_PATH = PROJECT_ROOT / PATHS_CONFIG.get("documents", "documents")
INDEX_PATH = PROJECT_ROOT / PATHS_CONFIG.get("index", "faiss_index")
PROMPT_PATH = PROJECT_ROOT / PATHS_CONFIG.get("prompts", "prompts/customer_care.txt") 
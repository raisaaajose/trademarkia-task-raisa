import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Model Config
    N_CLUSTERS: int = int(os.getenv("N_CLUSTERS", 14))
    MODEL_NAME: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Paths
    DATA_DIR: str = os.getenv("DATA_DIR", "data")
    GMM_PATH: str = os.path.join(DATA_DIR, "gmm_model.pkl")
    VECTOR_DB_PATH: str = os.path.join(DATA_DIR, "vector_db.faiss")

    # Cache
    CACHE_THRESHOLD: float = float(os.getenv("CACHE_THRESHOLD", 0.70))
    CACHE_CAPACITY: int = int(os.getenv("CACHE_CAPACITY", 100))

    T: float = float(os.getenv("T", 3.5))


config = Settings()

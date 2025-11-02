import os
from dotenv import load_dotenv
load_dotenv()

HANDBOOK_URL = os.getenv("HANDBOOK_URL")
PUBLIC_SAFETY_PHONE = os.getenv("PUBLIC_SAFETY_PHONE", "911")
CONDUCT_EMAIL = os.getenv("CONDUCT_EMAIL", "stconduct@hartford.edu")

# --- Choose engines ---
USE_GEMINI_EMBEDS = True      # use Gemini text-embedding-004
USE_GEMINI_LLM = True         # use Gemini 1.5 for answers
DB_PATH = "./.conduct_db"

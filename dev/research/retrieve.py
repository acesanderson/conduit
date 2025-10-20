from openai import OpenAI
import os
import time
from rich.console import Console
from pathlib import Path
import shelve

QUERY_STRING = (Path(__file__).parent / "query_string.jinja2").read_text()

with shelve.open("research_shelve.db") as db:
    response = db[QUERY_STRING]

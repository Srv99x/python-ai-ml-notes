"""
convert_to_notebooks.py
Batch converts all .py files in your Python folder into .ipynb files using Groq API.

SETUP:
1. python -m pip install groq nbformat
2. PowerShell: $env:GROQ_API_KEY="your_key_here"
3. python convert_to_notebooks.py
"""

import os
import re
import json
import time
import nbformat
from groq import Groq

# --- CONFIG ------------------------------------------------------------------

API_KEY = os.environ.get("GROQ_API_KEY", "YOUR_API_KEY_HERE")
PYTHON_FOLDER = r"C:\Users\SOURAV\Desktop\Python"
SKIP_FOLDERS = {"venv", ".vscode", "__pycache__"}
SKIP_FILES = {"convert_to_notebooks.py"}
DELAY_BETWEEN_FILES = 5

# -----------------------------------------------------------------------------

client = Groq(api_key=API_KEY)

SYSTEM_PROMPT = """You are an expert Python educator creating Jupyter Notebook content for a 2nd-year B.Tech CSE (AI & DS) student learning Python for AI and Data Science.

Convert the Python file into a structured Jupyter Notebook JSON.

Output ONLY valid raw JSON. No markdown fences, no explanation, no preamble. Just the JSON.
The JSON must follow the nbformat 4.4 standard exactly.

Cell structure rules:
1. First cell: Markdown - topic heading (##) + 2-3 line overview of what this file covers
2. For EACH concept/code block in the file:
   a. Markdown cell BEFORE code: what it is, why it's used, syntax pattern (concise, no fluff)
   b. Code cell: original code, cleaned up if messy, with inline comments where helpful
   c. Markdown cell AFTER code: what the output means, 1 common mistake, 1 quick tip
3. Last cell: Markdown - "## Key Takeaways" with bullet points of every concept covered
4. Language: simple and direct. Written for a beginner revising for AI/DS.
5. Do NOT skip any concept from the original file.

JSON structure:
{
 "nbformat": 4,
 "nbformat_minor": 5,
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.11.0"}
 },
 "cells": [
  {
   "cell_type": "markdown",
   "id": "cell-001",
   "metadata": {},
   "source": ["# Topic Title\\n", "Overview here."]
  },
  {
   "cell_type": "code",
   "id": "cell-002",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": ["print('hello')"]
  }
 ]
}"""

def read_py_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def clean_json(raw):
    # Strip markdown fences if Groq added them
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1])
    # Fix invalid escape sequences
    raw = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw)
    return raw

def convert_file_to_notebook(py_content, filename):
    print("    Calling Groq API...")

    user_prompt = "Convert this Python file '" + filename + "' into a Jupyter Notebook JSON:\n\n```python\n" + py_content + "\n```"

    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=8096
            )

            raw = response.choices[0].message.content.strip()
            raw = clean_json(raw)
            return json.loads(raw, strict=False)

        except json.JSONDecodeError:
            raise
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower() or "quota" in err.lower():
                wait = 30 * (attempt + 1)
                print("    Rate limited. Waiting " + str(wait) + "s (attempt " + str(attempt + 1) + "/5)...")
                time.sleep(wait)
            else:
                raise

    raise Exception("Failed after 5 retries")

def save_notebook(nb_json, output_path):
    nb = nbformat.from_dict(nb_json)
    nbformat.validate(nb)
    with open(output_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

def get_all_py_files(folder):
    py_files = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if d not in SKIP_FOLDERS]
        for file in files:
            if file.endswith(".py"):
                py_files.append(os.path.join(root, file))
    return py_files

def main():
    print("=" * 60)
    print("  Notebook Converter (Groq)")
    print("=" * 60)
    print()

    if not os.path.exists(PYTHON_FOLDER):
        print("ERROR: Folder not found: " + PYTHON_FOLDER)
        return

    py_files = get_all_py_files(PYTHON_FOLDER)
    total = len(py_files)

    if total == 0:
        print("No .py files found. Check PYTHON_FOLDER path.")
        return

    print("Found " + str(total) + " Python files to convert.")
    print()

    success = 0
    failed = []

    for i, py_path in enumerate(py_files, 1):
        filename = os.path.basename(py_path)
        output_path = py_path.replace(".py", ".ipynb")

        if filename in SKIP_FILES:
            print("[" + str(i) + "/" + str(total) + "] SKIP (script): " + filename)
            continue

        if os.path.exists(output_path):
            print("[" + str(i) + "/" + str(total) + "] SKIP (exists): " + filename)
            continue

        print("[" + str(i) + "/" + str(total) + "] Converting: " + filename)

        try:
            py_content = read_py_file(py_path)

            if not py_content.strip():
                print("    Skipped (empty file)")
                continue

            nb_json = convert_file_to_notebook(py_content, filename)
            save_notebook(nb_json, output_path)
            print("    DONE: " + filename.replace(".py", ".ipynb"))
            success += 1

        except json.JSONDecodeError as e:
            print("    FAILED (JSON error): " + str(e))
            failed.append(filename)
        except Exception as e:
            print("    FAILED: " + str(e))
            failed.append(filename)

        if i < total:
            time.sleep(DELAY_BETWEEN_FILES)

    print()
    print("=" * 60)
    print("Done. " + str(success) + " converted, " + str(len(failed)) + " failed.")
    if failed:
        print("Failed files:")
        for f in failed:
            print("  - " + f)
    print("=" * 60)

if __name__ == "__main__":
    main()
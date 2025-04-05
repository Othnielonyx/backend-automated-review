from fastapi import FastAPI, File, UploadFile
import subprocess
import os
import json
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Adjust frontend port if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tip/Resource suggestions
resources = {
    "C0114": "https://realpython.com/documenting-python-code/",
    "C0116": "https://realpython.com/documenting-python-code/#documenting-functions",
    "W0622": "https://peps.python.org/pep-0008/#naming-conventions",
    "R0913": "https://sourcemaking.com/refactoring/smells/long-parameter-list",
    "W0611": "https://realpython.com/python-import/",
    "C0301": "https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html#line-length",
    "R0201": "https://realpython.com/instance-class-and-static-methods-demystified/",
}

def translate_pylint_message(issue):
    translations = {
        "C0114": "Your script does not have a module-level docstring. Add a brief description at the top of your file.",
        "C0116": "The function '{}' is missing a docstring. Add a short comment inside triple quotes to describe what it does.",
        "W0622": "You're using a built-in name '{}'. Rename it to avoid conflicts, e.g., use 'total' instead of 'sum'.",
        "R0913": "The function '{}' has too many parameters. Consider reducing them for better readability and maintainability.",
        "W0611": "The import '{}' is not being used. Remove it to clean up your code.",
        "C0301": "Line too long ({} characters). Try breaking it into multiple lines to improve readability.",
        "R0201": "Method '{}' could be a static method since it doesn’t use 'self'."
    }

    message_id = issue["message-id"]
    obj_name = issue.get("obj", "")

    if message_id in translations:
        return translations[message_id].format(obj_name)

    return issue["message"]

def generate_human_readable_report(pylint_output):
    try:
        report = json.loads(pylint_output)
        formatted_messages = []
        for issue in report:
            formatted_messages.append({
                "line": issue["line"],
                "message": translate_pylint_message(issue),
                "tip": resources.get(issue["message-id"], None)
            })
        return formatted_messages if formatted_messages else [{"message": "No issues found. ✅"}]
    except json.JSONDecodeError:
        return [{"message": "Failed to parse Pylint output."}]

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        os.makedirs("temp", exist_ok=True)
        file_location = f"temp/{file.filename}"

        with open(file_location, "wb") as f:
            f.write(await file.read())

        result = subprocess.run(
            ["pylint", file_location, "--output-format=json"],
            capture_output=True,
            text=True
        )

        score_output = subprocess.run(
            ["pylint", file_location],
            capture_output=True,
            text=True
        )

        raw_score_text = score_output.stdout
        score_line = next((line for line in raw_score_text.splitlines() if "Your code has been rated at" in line), None)
        score = None
        if score_line:
            score = float(score_line.split("/")[0].split()[-1])

        formatted_output = generate_human_readable_report(result.stdout or "[]")

        # Count problems only if they are not "No issues found"
        if formatted_output and formatted_output[0].get("message") == "No issues found. ✅":
            problem_count = 0
        else:
            problem_count = len(formatted_output)

        os.remove(file_location)

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "File uploaded and analyzed.",
                "output": formatted_output,
                "score": score,
                "problem_count": problem_count  # ✅ Added here!
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)})

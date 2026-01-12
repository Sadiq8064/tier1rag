import os
import time
import shutil
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from google import genai
from google.genai import types
from pydantic import BaseModel

app = FastAPI(title="Project Antigravity - Gemini RAG Backend")

# Initialize Gemini Client
API_KEY = os.getenv("GEMINI_API_KEY", "AQ.Ab8RN6L-CeSoymIyq6Xofek5Uoymyyh-OCIRtAmflJksXuepIg")
client = genai.Client(
    api_key=API_KEY,
    http_options={'api_version': 'v1beta'} # This is CRITICAL for File Search API Keys
)

# --- Models ---
class StoreCreate(BaseModel):
    display_name: str

class QueryRequest(BaseModel):
    store_name: str  # Format: fileSearchStores/xyz
    question: str
    model: str = "gemini-2.5-flash"

# --- Endpoints ---

@app.post("/stores")
async def create_store(data: StoreCreate):
    """Creates a new File Search Store."""
    try:
        store = client.file_search_stores.create(
            config={'display_name': data.display_name}
        )
        return {"status": "success", "store_name": store.name, "display_name": store.display_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stores")
async def list_stores():
    """Lists all available stores."""
    stores = [s for s in client.file_search_stores.list()]
    return {"stores": stores}

@app.post("/stores/upload")
async def upload_document(
    store_name: str, 
    display_name: str,
    file: UploadFile = File(...)
):
    """Uploads and indexes a file into a specific store."""
    # Temporarily save file to disk for SDK to read
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        operation = client.file_search_stores.upload_to_file_search_store(
            file=temp_path,
            file_search_store_name=store_name,
            config={'display_name': display_name}
        )
        
        # Wait for indexing to complete (Long Running Operation)
        while not operation.done:
            time.sleep(2)
            operation = client.operations.get(operation)

        return {"status": "indexed", "file": display_name, "store": store_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.delete("/stores/{store_id}")
async def delete_store(store_id: str):
    """Deletes an entire store (Pass store_id without the 'fileSearchStores/' prefix)."""
    full_name = f"fileSearchStores/{store_id}"
    try:
        client.file_search_stores.delete(name=full_name, config={'force': True})
        return {"status": "deleted", "store": full_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(request: QueryRequest):
    """Queries the store and returns grounded response + citations."""
    try:
        response = client.models.generate_content(
            model=request.model,
            contents=request.question,
            config=types.GenerateContentConfig(
                tools=[
                    types.Tool(
                        file_search=types.FileSearch(
                            file_search_store_names=[request.store_name]
                        )
                    )
                ]
            )
        )

        # Extracting grounding metadata for verification
        grounding_data = None
        if response.candidates and response.candidates[0].grounding_metadata:
            grounding_data = response.candidates[0].grounding_metadata

        return {
            "answer": response.text,
            "grounding_metadata": grounding_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

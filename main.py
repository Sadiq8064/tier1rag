import os
import time
import shutil
import json
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from google import genai
from google.genai import types
from pydantic import BaseModel

app = FastAPI(title="Project Antigravity - Gemini RAG Backend")

# 1. Credential Setup
# You can either save your JSON content to a file or load it directly.
# For security on Render, it's best to save the JSON content as an Environment Variable
SERVICE_ACCOUNT_INFO = {
  "type": "service_account",
  "project_id": "quicknotes-24e44",
  "private_key_id": "5e35d66c41b67c1d02af77efa1ca84817c41cee8",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDLLUXVSTfS7q4r\nr83qm5isapjK6M8Xl5lR4vyB1nPJLaRGj2dtxIruh/paj/jJilkEp0onf7cEWj5I\n1/JMpsn6b9OeCZUzgF+J2erTtYqlbIsRB+u7UHqNvjJOnLv7Fli6k+DCDpa3xZNN\nqjUmVZ4k2NZYF2pKhd+AVceRFxC/vYMI2odM23+x+nIp3+bc5cgQ5D2XE2VxnV2X\nfYeFpI1jG6PyezBTiDaLMvbCnhFee/Bom3vzaBH10ZrB5hdeFp3FKJkaWSIUqvax\nLGgsqtIMeMpAIuyHEfiIkDblCG5+SZlmly6TbIrww7dfXHa3h35owK6AQyT0VDNl\nn2NsY56FAgMBAAECggEAGnzKoQLRwsTNXbJRKNfjwFKInHyxBt/Yzpqbav0qDMek\n+3KVkXNb3hCfHEWvBu3rgKujLgwR2tY9pCHMRkmWL6PQKvG6iu4eRmqWQ8cOjgCy\nY+QsS2KzX3LfZxfleoHldl7HSzU7WTY/kmfxVDqO3+SJk7lrE3ppcqHRicI/3jxw\nUQsB5RZ/Mxwmv0wWs+fVTlqaEju2n7I3WWQ+tY/NAhb6YNMIF4O7KkCE1h8evmsN\nWJIiBarN2dxsDlGN/mdHvlhUoNAODxVpOqckzu2vdidY0mdplYCyEqmicLTAk5BE\nJZsEAqSxY7A1SMjIYeA6+4WDRRcvhuNN4Q75fYJoDwKBgQD83zpapQQskvnYsmqO\ndYSMD7RJ4JbF1U5TILIAn14ham8Bpp/3OmRLpT8PE2S3itJF8FeJlOtyAhdo5F6E\nYQ2hrSTqlWsEdT2t1V/yMaLazv4oPXRNRFU16Hyg1FR5dY9I/VimOsFkbi4bz6Lb\nvHVEc/uux65rgrlQ1Ns4v0DNwwKBgQDNsKy+7srEOo2LmvCMeZTTFo8MF8DrMf6O\n/Ny7leI6WxBOqkch/YEus0jaU6IOW8gJHVkRZuL+STlTarZphAVR4SZQnZG/i1uE\nX5mt9F1QZngm0czcOmyXdSnANpISBNou/v7AqtOMu53F/9rjsk6uZnmJu6xpZmny\nfXHG0I42FwKBgF1f8KRYGtp6y1eBSmegbXlqsyA0lngm8+0uPYyNTKz6KFNru9YG\n8dIdCtP+TUqZMFwzC5/6JZbLvuk0qtInJGl3DhKxafsTb9so120PdxtlI+SoDLIb\ncXuehaa9wRO4nVhOdNWtDIyRQuVyHqkVRhgo298GTRnWA6gdcXXa6YsRAoGBAL3S\nW0mFVAkNxEze+zmYhnjb672MOlSTecn5n7daFkImggEm8ahzrAEuIYCjB/5aQ1vu\nAqGlorxrVQUfiSINXV93+gURtUzgzd923nuD9Y3aUu34Vieznz2TOamScvFAGx6d\n4vppa/wHtQ3iFd5mUmhuV6F9WhqXEhVgIV0KYQOXAoGBALQCsdlP4mjVmwYWeEVu\niE8eLUuCd/OCqzCojv7tJOW/+3WLVG3R8U8uakqFcCsa1TsJk91NWy/f8QWR6wxs\nWuuVqpMTpE8KUVq7LlN5ItUkC9iviAq0K4fLtEz3uuk9yIuW9UpNQa0rTnyGuXjQ\ntz9sQ/SP1koTywhiZxG/MHi6\n-----END PRIVATE KEY-----\n",
  "client_email": "gemini-file-manager@quicknotes-24e44.iam.gserviceaccount.com",
  "client_id": "106745777276684281363",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/gemini-file-manager%40quicknotes-24e44.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}

# 2. Initialize Gemini Client using Vertex AI mode (Required for Service Accounts)
# This will handle the OAuth2 handshake automatically.
client = genai.Client(
    vertexai=True,
    project="quicknotes-24e44",
    location="us-central1", # You can change this to your preferred region
    credentials=SERVICE_ACCOUNT_INFO
)

# --- Models ---
class StoreCreate(BaseModel):
    display_name: str

class QueryRequest(BaseModel):
    store_name: str
    question: str
    model: str = "gemini-2.0-flash" # Use supported models for File Search

# --- Endpoints ---

@app.get("/")
async def health_check():
    return {"status": "online", "project": "Project Antigravity"}

@app.post("/stores")
async def create_store(data: StoreCreate):
    try:
        # File Search stores in Vertex mode are managed via the retriever service
        store = client.file_search_stores.create(
            config={'display_name': data.display_name}
        )
        return {"status": "success", "store_name": store.name, "display_name": store.display_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Store Creation Failed: {str(e)}")

@app.get("/stores")
async def list_stores():
    try:
        stores = [s for s in client.file_search_stores.list()]
        return {"stores": stores}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stores/upload")
async def upload_document(store_name: str, display_name: str, file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        operation = client.file_search_stores.upload_to_file_search_store(
            file=temp_path,
            file_search_store_name=store_name,
            config={'display_name': display_name}
        )
        
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
    # Store ID usually follows the format: projects/{project}/locations/{loc}/fileSearchStores/{id}
    # In Vertex mode, use the full resource name
    try:
        client.file_search_stores.delete(name=store_id, config={'force': True})
        return {"status": "deleted", "store": store_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(request: QueryRequest):
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
    # Render provides the PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials


from app.utils.auth import get_user, verify_password
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.rag import answer_question


app = FastAPI()
security = HTTPBasic()



# Authentication dependency
def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    user = get_user(credentials.username)
    if not user or not verify_password(credentials.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"username": user["username"], "role": user["role"]}


# Login endpoint
@app.get("/login")
def login(user=Depends(authenticate)):
    return {"message": f"Welcome {user['username']}!", "role": user["role"]}


# Protected test endpoint
@app.get("/test")
def test(user=Depends(authenticate)):
    return {"message": f"Hello {user['username']}! You can now chat.", "role": user["role"]}


# Protected chat endpoint
@app.post("/chat")
def query(request: ChatRequest, user=Depends(authenticate)) -> ChatResponse:
    """Run the RBAC-filtered RAG pipeline for the logged-in user."""
    answer = answer_question(request.message, user["role"])
    return ChatResponse(answer=answer, department=user["role"])
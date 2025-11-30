# eliyon_api.py

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
    Depends,
    status,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from ultralytics import YOLO
import numpy as np
import cv2
import torch
import os
import uuid

from pymongo import MongoClient, ASCENDING
from bson import ObjectId
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from pydantic import BaseModel, EmailStr, Field
from fastapi.security import OAuth2PasswordBearer

# ======================= CONFIG =======================

# --- Model paths (same as babytest.py) ---
DOG_EMOTION_PATH = r"C:\Users\sanu\Desktop\emotion\runs\dog_emotion_cls\weights\best.pt"
DOG_HEALTH_PATH  = r"C:\Users\sanu\Desktop\emotion\dog_disease_cls_runs\yolov8s_disease_v1_gpu_fixed\weights\best.pt"

CAT_EMOTION_PATH = r"C:\Users\sanu\Desktop\emotion\cat_emotion_cls_runs\yolov8s_cat_emotion_v1_gpu\weights\best.pt"
CAT_HEALTH_PATH  = r"C:\Users\sanu\Desktop\emotion\cat_disease_cls_runs\yolov8s_cat_disease_v1_gpu\weights\best.pt"

BABY_EMOTION_PATH     = r"C:\Users\sanu\Desktop\emotion\baby_emotion_cls_runs\yolov8s_baby_emotion_v1_gpu\weights\best.pt"
BABY_SKIN_MODEL_PATH  = r"C:\Users\sanu\Desktop\emotion\runs\classify\baby_skin_disease_model13\weights\best.pt"
BABY_EAR_MODEL_PATH   = r"C:\Users\sanu\Desktop\emotion\baby_disease_cls_runs\yolov8s_baby_ear_disease_v1\weights\best.pt"

MILD_LOW  = 0.25
MILD_HIGH = 0.55
EAR_CLASSES = {"otitis_externa", "otitis_media", "perforation", "wax"}

# --- Upload directory for saving images ---
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- MongoDB config ---
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
client = MongoClient(MONGO_URL)
db = client["eliyon_db"]
users_col = db["users"]
scans_col = db["scans"]

# Ensure email is unique
users_col.create_index([("email", ASCENDING)], unique=True)
# Optional: index for faster history queries
scans_col.create_index([("user_id", ASCENDING), ("created_at", ASCENDING)])

# --- Auth / JWT config ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

SECRET_KEY = os.getenv("ELIYON_SECRET_KEY", "super-secret-key-change-this")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 1 day

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# ======================= FASTAPI APP =======================

app = FastAPI(title="Eliyon API")

# CORS (allow frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # for dev; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded images at /uploads/...
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# ======================= AUTH HELPERS =======================

def _normalize_password(password: str) -> str:
    """
    bcrypt only supports up to 72 bytes.
    This trims to 72 bytes in UTF-8 and decodes back.
    """
    b = password.encode("utf-8")[:72]
    return b.decode("utf-8", "ignore")

def hash_password(password: str) -> str:
    norm = _normalize_password(password)
    return pwd_context.hash(norm)

def verify_password(plain: str, hashed: str) -> bool:
    norm = _normalize_password(plain)
    return pwd_context.verify(norm, hashed)

def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str | None = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    try:
        user = users_col.find_one({"_id": ObjectId(user_id)})
    except Exception:
        user = None

    if user is None:
        raise credentials_exception
    return user

# ======================= AUTH MODELS & ENDPOINTS =======================

class UserCreate(BaseModel):
    email: EmailStr
    # Enforce sane password length to avoid bcrypt issues & abuse
    password: str = Field(min_length=8, max_length=64)
    name: str | None = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=64)

@app.post("/auth/signup")
def signup(user: UserCreate):
    # Check if email already used
    if users_col.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed = hash_password(user.password)
    doc = {
        "email": user.email,
        "password_hash": hashed,
        "name": user.name,
        "created_at": datetime.utcnow(),
    }
    result = users_col.insert_one(doc)

    # ✅ Auto-generate token so frontend doesn't ask user to login again
    token = create_access_token({"sub": str(result.inserted_id)})

    return {
        "message": "User created",
        "user_id": str(result.inserted_id),
        "access_token": token,
        "token_type": "bearer",
        "email": user.email,
        "name": user.name,
    }

@app.post("/auth/login")
def login(creds: UserLogin):
    user = users_col.find_one({"email": creds.email})
    if not user or not verify_password(creds.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token({"sub": str(user["_id"])})
    return {"access_token": token, "token_type": "bearer"}

@app.get("/auth/me")
def get_me(current_user: dict = Depends(get_current_user)):
    return {
        "id": str(current_user["_id"]),
        "email": current_user["email"],
        "name": current_user.get("name"),
        "created_at": current_user.get("created_at"),
    }

# ======================= YOLO MODEL LOADING =======================

print("Loading Eliyon models for API...")
try:
    dog_emotion_model = YOLO(DOG_EMOTION_PATH)
    dog_health_model  = YOLO(DOG_HEALTH_PATH)

    cat_emotion_model = YOLO(CAT_EMOTION_PATH)
    cat_health_model  = YOLO(CAT_HEALTH_PATH)

    baby_emotion_model = YOLO(BABY_EMOTION_PATH)
    baby_skin_model    = YOLO(BABY_SKIN_MODEL_PATH)
    baby_ear_model     = YOLO(BABY_EAR_MODEL_PATH)
except Exception as e:
    print("❌ Error loading one or more models.")
    raise e

print("✅ Models loaded for API.")

# ======================= HELPERS (YOLO + LOGIC) =======================

def get_top_from_result(res):
    """
    Returns (class_name, confidence_float).
    """
    try:
        probs = getattr(res, "probs", None)
        if probs is None:
            return None, 0.0

        p = getattr(probs, "data", None)
        if isinstance(p, torch.Tensor):
            if p.ndim == 2:
                p = p[0]
            idx = int(torch.argmax(p).item())
            conf = float(p[idx].item())
            name = res.names[idx]
            return str(name), conf
    except Exception:
        pass

    # fallback: use top1 if present
    try:
        idx = int(res.probs.top1)
        conf = float(res.probs[idx])
        name = res.names[idx]
        return str(name), conf
    except Exception:
        pass

    return "unknown", 0.0


def baby_health_decision_api(img):
    """
    Baby health fusion for API.
    img: numpy array (BGR) from cv2.
    Returns final_label, reason, breakdown(dict).
    """
    r_skin = baby_skin_model(img)[0]
    r_ear  = baby_ear_model(img)[0]

    skin_class, skin_conf = get_top_from_result(r_skin)
    ear_class, ear_conf   = get_top_from_result(r_ear)

    skin_class = skin_class or "unknown"
    ear_class  = ear_class or "unknown"

    final = None
    reason = ""

    if skin_class.lower() in ["healthy", "normal"]:
        if ear_class in EAR_CLASSES:
            if ear_conf > MILD_HIGH:
                final = ear_class
                reason = f"Strong ear disease predicted by ear model ({ear_conf:.2f})"
            elif MILD_LOW <= ear_conf <= MILD_HIGH:
                final = f"Mild {ear_class}"
                reason = f"Early/mild ear concern (conf {ear_conf:.2f})"
            else:
                final = "Healthy"
                reason = "Both skin model and ear model agree on health (low ear confidence)"
        else:
            final = "Healthy"
            reason = "No skin or ear disease detected"
    else:
        # skin disease
        if ear_class in EAR_CLASSES:
            if ear_conf > MILD_HIGH:
                final = f"{skin_class} + {ear_class}"
                reason = "Skin disease + strong ear disease"
            elif MILD_LOW <= ear_conf <= MILD_HIGH:
                final = f"{skin_class} + mild {ear_class}"
                reason = "Skin disease and mild ear concern"
            else:
                final = skin_class
                reason = "Skin disease detected; ear model low confidence"
        else:
            final = skin_class
            reason = "Skin disease detected; ear model normal"

    breakdown = {
        "skin_model": [skin_class, float(skin_conf)],
        "ear_model" : [ear_class, float(ear_conf)],
    }
    return final, reason, breakdown


def emotion_explanation(species, label):
    """Return (explanation, severity) similar to your JS mock logic."""
    l = (label or "").lower()

    if species == "baby":
        if l in ["cry", "crying", "angry"]:
            return (
                "Baby shows strong discomfort. Consider doing a health scan if you notice fever, rash, or persistent crying.",
                "moderate",
            )
        else:
            return (
                "Baby appears emotionally stable in this moment. Still keep an eye on any physical symptoms.",
                "clear",
            )

    if species == "dog":
        if l in ["sad", "anxious"]:
            return (
                "Dog may be in pain or stressed. Combine this with a health scan if behavior continues.",
                "mild",
            )
        else:
            return (
                "Dog seems emotionally okay in the captured moment.",
                "clear",
            )

    # cat
    if l in ["annoyed", "scared"]:
        return (
            "Cat looks uncomfortable. Sudden changes in mood can be linked to pain or illness.",
            "mild",
        )
    else:
        return (
            "Cat appears reasonably relaxed. Watch for changes in appetite or grooming.",
            "clear",
        )


def health_explanation(label, conf, reason=None):
    """
    Return (explanation, severity) similar to JS logic, using label text.
    """
    l = (label or "").lower()

    if l == "healthy" or "healthy" in l:
        return (
            "No obvious visual disease patterns detected. This does not replace a medical/veterinary exam.",
            "clear",
        )

    if "mild" in l:
        return (
            "Visual patterns show mild risk. Watch symptoms and consider seeking professional advice if they persist.",
            "mild",
        )

    if "unlikely" in l or "allergy" in l:
        return (
            "Some indicators appear, but not strongly. This is a hint, not a final diagnosis.",
            "moderate",
        )

    # default = severe
    base = "Clear disease-like visual patterns detected. We strongly recommend seeing a doctor or vet."
    if reason:
        base = reason + " " + base
    return base.strip(), "severe"

# ======================= MAIN PREDICT ENDPOINT =======================

@app.post("/predict")
async def predict(
    request: Request,
    species: str = Form(...),
    mode: str = Form(...),
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """
    Main prediction endpoint.
    - Requires Authorization: Bearer <token> (JWT)
    - Saves uploaded image to disk
    - Runs YOLO models
    - Stores scan history in MongoDB
    """
    species = species.lower()
    mode = mode.lower()

    if species not in {"baby", "dog", "cat"}:
        raise HTTPException(status_code=400, detail="Invalid species")
    if mode not in {"emotion", "health"}:
        raise HTTPException(status_code=400, detail="Invalid mode")

    # --- 1) Read file bytes & save to disk ---
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    orig_ext = os.path.splitext(file.filename)[1] or ".jpg"
    unique_name = f"{uuid.uuid4().hex}{orig_ext}"
    save_path = os.path.join(UPLOAD_DIR, unique_name)
    with open(save_path, "wb") as f:
        f.write(contents)

    base_url = str(request.base_url).rstrip("/")
    image_url = f"{base_url}/uploads/{unique_name}"

    # --- 2) Run models ---
    result_label = ""
    result_conf = 0.0
    result_severity = "clear"
    ai_insight = ""
    breakdown = None

    # ------- EMOTION -------
    if mode == "emotion":
        if species == "baby":
            res = baby_emotion_model(img)[0]
        elif species == "dog":
            res = dog_emotion_model(img)[0]
        else:
            res = cat_emotion_model(img)[0]

        label, conf = get_top_from_result(res)
        label = label or "Unknown"
        explanation, severity = emotion_explanation(species, label)

        result_label = label
        result_conf = float(conf)
        result_severity = severity
        ai_insight = explanation

        response = {
            "species": species,
            "mode": mode,
            "label": result_label,
            "confidence": result_conf,
            "severity": result_severity,
            "ai_insight": ai_insight,
            "image_url": image_url,
        }

    # ------- HEALTH -------
    else:  # mode == "health"
        if species == "baby":
            final_label, reason, breakdown = baby_health_decision_api(img)
            skin_conf = breakdown["skin_model"][1]
            ear_conf  = breakdown["ear_model"][1]
            conf = max(skin_conf, ear_conf)
            explanation, severity = health_explanation(final_label, conf, reason)

            result_label = final_label
            result_conf = float(conf)
            result_severity = severity
            ai_insight = explanation

            response = {
                "species": species,
                "mode": mode,
                "label": result_label,
                "confidence": result_conf,
                "severity": result_severity,
                "ai_insight": ai_insight,
                "breakdown": breakdown,
                "image_url": image_url,
            }
        else:
            # dog/cat health is single model
            if species == "dog":
                res = dog_health_model(img)[0]
            else:
                res = cat_health_model(img)[0]

            label, conf = get_top_from_result(res)
            label = label or "Unknown"
            explanation, severity = health_explanation(label, conf)

            result_label = label
            result_conf = float(conf)
            result_severity = severity
            ai_insight = explanation

            response = {
                "species": species,
                "mode": mode,
                "label": result_label,
                "confidence": result_conf,
                "severity": result_severity,
                "ai_insight": ai_insight,
                "image_url": image_url,
            }

    # --- 3) Save scan history in MongoDB ---
    scan_doc = {
        "user_id": current_user["_id"],
        "created_at": datetime.utcnow(),
        "species": species,
        "mode": mode,
        "label": result_label,
        "confidence": result_conf,
        "severity": result_severity,
        "ai_insight": ai_insight,
        "image_filename": unique_name,
    }
    if breakdown is not None:
        scan_doc["breakdown"] = breakdown

    scans_col.insert_one(scan_doc)

    # --- 4) Return response JSON ---
    return response

# ======================= HISTORY ENDPOINT =======================

@app.get("/history")
def get_history(request: Request, current_user: dict = Depends(get_current_user)):
    """
    Return all scan history for the logged-in user,
    newest first, including image URLs.
    """
    docs = scans_col.find(
        {"user_id": current_user["_id"]},
        sort=[("created_at", -1)]
    )

    base_url = str(request.base_url).rstrip("/")

    history = []
    for d in docs:
        created_at = d.get("created_at")
        if isinstance(created_at, datetime):
            created_str = created_at.isoformat()
        else:
            created_str = None

        item = {
            "id": str(d["_id"]),
            "created_at": created_str,
            "species": d.get("species"),
            "mode": d.get("mode"),
            "label": d.get("label"),
            "confidence": d.get("confidence"),
            "severity": d.get("severity"),
            "ai_insight": d.get("ai_insight", ""),
        }
        img_name = d.get("image_filename")
        if img_name:
            item["image_url"] = f"{base_url}/uploads/{img_name}"
        else:
            item["image_url"] = None

        if "breakdown" in d:
            item["breakdown"] = d["breakdown"]

        history.append(item)

    return history

# ======================= FRONTEND ROUTES (SERVE WEBSITE) =======================

@app.get("/", include_in_schema=False)
def serve_index():
    # Make sure index.html is in same folder as this file
    return FileResponse("index.html")


@app.get("/style.css", include_in_schema=False)
def serve_css():
    return FileResponse("style.css", media_type="text/css")


@app.get("/script.js", include_in_schema=False)
def serve_js():
    return FileResponse("script.js", media_type="application/javascript")


import cv2
import numpy as np
import os
import shutil
import tempfile
import base64
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse 
from scipy.spatial.distance import cdist
from scipy.stats import zscore

# --- LOCAL IMPORTS ---
import video_process
import database
from visual_coach import VisualChain

app = FastAPI()

# --- CORS CONFIGURATION ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for flexibility, or use ["http://localhost:3000"] without trailing slash
    allow_credentials=True,
    allow_methods=["*"], # Must be "*" or specific methods like "POST", "GET"
    allow_headers=["*"], # Must be "*" or specific headers
)

# Initialize The Coach
coach = VisualChain()

# --- CONSTANTS ---
TARGET_FRAMES = 100     # Number of frames to extract
ERROR_THRESHOLD = 0.8   # Z-score distance threshold
MAX_ERROR_FRAMES = 10    # Max number of distinct error events to report
FRAME_CLUSTER_GAP = 10  # Frames within this distance are considered the same "event"

# --- HELPER FUNCTIONS ---
def image_to_base64(img_array):
    _, buffer = cv2.imencode('.jpg', img_array)
    return base64.b64encode(buffer).decode('utf-8')

# --- Pydantic Models ---
from pydantic import BaseModel
from typing import List, Optional, Any

class ChatRequest(BaseModel):
    message: str
    history: List[Any] = []

class LoginRequest(BaseModel):
    email: str
    password: str

class SignupRequest(BaseModel):
    email: str
    password: str
    attributes: Optional[dict] = {}

# ROUTES 

@app.post("/signup")
async def signup(request: SignupRequest):
    success = database.new_user(request.email, request.password, request.attributes)
    if not success:
        raise HTTPException(status_code=400, detail="User already exists")
    return {"status": "success", "message": "User created"}

@app.post("/login")
async def login(request: LoginRequest):
    if database.verify_user(request.email, request.password):
        return {"status": "success", "message": "Login successful"}
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/history")
async def get_history(email: str):
    history = database.fetch_user_history(email)
    return {"history": history} 

# @app.get("/")
# async def read_index():
#     if os.path.exists("index.html"):
#         return FileResponse("index.html")
#     return {"message": "index.html not found. Please upload your frontend file."}


@app.post("/analyze_movement")
async def analyze_movement(
    trainer_video: UploadFile = File(...), 
    user_video: UploadFile = File(...),
    exercise_name: str = "Exercise",
    email: str = "guest"
):
    t_path = ""
    u_path = ""
    
    try:
        # 1. Save uploaded videos temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as t_tmp:
            shutil.copyfileobj(trainer_video.file, t_tmp)
            t_path = t_tmp.name
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as u_tmp:
            shutil.copyfileobj(user_video.file, u_tmp)
            u_path = u_tmp.name

        # 2. EXTRACT SKELETON DATA
        print(f"Processing Trainer Video: {t_path}")
        t_data = video_process.process_video_path(t_path, TARGET_FRAMES)
        
        print(f"Processing User Video: {u_path}")
        u_data = video_process.process_video_path(u_path, TARGET_FRAMES)

        # Filter out frames where no person was detected
        t_data = [d for d in t_data if d['exists']]
        u_data = [d for d in u_data if d['exists']]

        if len(t_data) == 0 or len(u_data) == 0:
            raise HTTPException(status_code=400, detail="No person detected in one of the videos.")

        # 3. PREPARE FEATURE VECTORS FOR SYNCING
        t_feats_raw = np.array([d['features'] for d in t_data])
        u_feats_raw = np.array([d['features'] for d in u_data])

        # Normalize (Z-Score)
        t_feats_norm = np.nan_to_num(zscore(t_feats_raw, axis=0))
        u_feats_norm = np.nan_to_num(zscore(u_feats_raw, axis=0))

        # 4. SYNCHRONIZATION (Cost Matrix)
        print("Synchronizing videos...")
        dist_matrix = cdist(u_feats_norm, t_feats_norm, metric='cityblock')
        best_match_indices = np.argmin(dist_matrix, axis=1)

        # 5. IDENTIFY & CLUSTER ERRORS
        raw_error_candidates = []
        
        # 5a. Collect all frames that fail the threshold
        for u_idx, t_idx in enumerate(best_match_indices):
            cost = dist_matrix[u_idx, t_idx]
            if cost > ERROR_THRESHOLD:
                raw_error_candidates.append({
                    "cost": cost,
                    "u_idx": u_idx,
                    "t_idx": t_idx
                })

        # 5b. CLUSTERING LOGIC
        # If we have errors, group them by time so we don't spam similar frames
        unique_errors = []
        
        if raw_error_candidates:
            # Sort by user frame index (time)
            raw_error_candidates.sort(key=lambda x: x["u_idx"])
            
            # Initialize first cluster
            current_cluster = [raw_error_candidates[0]]
            
            for i in range(1, len(raw_error_candidates)):
                curr_frame = raw_error_candidates[i]
                prev_frame = current_cluster[0]
                
                # Check distance: If within gap, add to current cluster
                if (curr_frame["u_idx"] - prev_frame["u_idx"]) <= FRAME_CLUSTER_GAP:
                    current_cluster.append(curr_frame)
                else:
                    # Gap is too large, cluster ended. 
                    # Find the "worst" frame (highest cost) in the completed cluster
                    worst_frame_in_cluster = max(current_cluster, key=lambda x: x["cost"])
                    unique_errors.append(worst_frame_in_cluster)
                    
                    # Start new cluster
                    current_cluster = [curr_frame]
            
            # Don't forget the last cluster
            if current_cluster:
                worst_frame_in_cluster = max(current_cluster, key=lambda x: x["cost"])
                unique_errors.append(worst_frame_in_cluster)

        # 5c. Sort distinct events by error severity and take top N
        unique_errors.sort(key=lambda x: x["cost"], reverse=True)
        top_errors = unique_errors[:MAX_ERROR_FRAMES]

        print(f"Found {len(raw_error_candidates)} total bad frames.")
        print(f"Condensed into {len(unique_errors)} distinct error events.")
        print(f"Analyzing top {len(top_errors)} events.")

        # 6. GENERATE VISUAL FEEDBACK (VLM + LLM)
        results = []
        
        for item in top_errors:
            u_idx = item['u_idx']
            t_idx = item['t_idx']
            
            # Retrieve raw data
            u_frame_data = u_data[u_idx]
            t_frame_data = t_data[t_idx]
            
            # Draw skeletons
            u_img_skel = video_process.draw_skeleton_on_image(u_frame_data['image'], u_frame_data['kpts'])
            t_img_skel = video_process.draw_skeleton_on_image(t_frame_data['image'], t_frame_data['kpts'])
            
            # Convert to Base64
            u_b64 = image_to_base64(u_img_skel)
            t_b64 = image_to_base64(t_img_skel)
            
            # Call the AI Coach
            print(f"Requesting AI feedback for frame pair (User: {u_idx}, Trainer: {t_idx})...")
            ai_feedback = coach.analyze_images(u_b64, t_b64, exercise_name)
            
            results.append({
                "frame_id": int(u_idx),
                "error_score": round(item['cost'], 2),
                "feedback": ai_feedback,
                "user_image": u_b64, 
                "trainer_image": t_b64 
            })
            
            # 7. SAVE SESSION TO DATABASE
            # Extract meaningful mistakes (feedback) from results
            mistakes_list = [r['feedback'] for r in results]

            session_data = {
                'exercise_name': exercise_name,
                'mistakes': mistakes_list,
                'feedback': results[0]['feedback'] if results else "No specific feedback",
                'frames': [r['user_image'] for r in results],
                'summary': f"Analyzed {len(t_data)} frames.",
                'detailed_analysis': results
            }
            database.save_exercise_session(email, session_data)

            return {"status": "success", "analysis": results, "mistakes": mistakes_list}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

        if os.path.exists(t_path): os.remove(t_path)
        if os.path.exists(u_path): os.remove(u_path)

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        response = coach.chat_with_coach(request.message, request.history)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

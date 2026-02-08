
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load Models (Global Scope to load only once)
print(f"Loading AI Models on {DEVICE}...")
det_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
det_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365", device_map=DEVICE)

pose_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-plus-huge")
pose_model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-plus-huge", device_map=DEVICE)
print("Models loaded successfully.")

# COCO Keypoint Connections for Drawing
SKELETON_CONNECTIONS = [
    (5, 7), (7, 9),       # Left Arm
    (6, 8), (8, 10),      # Right Arm
    (5, 6),               # Shoulders
    (5, 11), (6, 12),     # Torso
    (11, 12),             # Hips
    (11, 13), (13, 15),   # Left Leg
    (12, 14), (14, 16)    # Right Leg
]

def calculate_angle_2d(a, b, c):
    """ Calculate angle between three points (a-b-c) in 2D space. """
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    norm_ba, norm_bc = np.linalg.norm(ba), np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0: return 0.0
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def draw_skeleton_on_image(image_array, kpts, confidence_threshold=0.3):
    """
    Draws the skeleton on a numpy image array using extracted keypoints.
    Used before sending to VLM.
    """
    img_copy = image_array.copy()
    
    # Draw Points
    for idx, (x, y) in enumerate(kpts):
        # We only draw main body joints (5 to 16)
        if 5 <= idx <= 16: 
            cv2.circle(img_copy, (int(x), int(y)), 4, (0, 255, 0), -1)

    # Draw Lines
    for start_idx, end_idx in SKELETON_CONNECTIONS:
        if start_idx < len(kpts) and end_idx < len(kpts):
            pt1 = (int(kpts[start_idx][0]), int(kpts[start_idx][1]))
            pt2 = (int(kpts[end_idx][0]), int(kpts[end_idx][1]))
            cv2.line(img_copy, pt1, pt2, (0, 0, 255), 2) # Red lines
            
    return img_copy

def get_kp(kpts, idx):
    if idx >= len(kpts): return np.array([0.0, 0.0])
    return kpts[idx].cpu().numpy()

def process_pose_batch(batch_images, batch_boxes, batch_indices, results_container):
    """ Runs VitPose on a batch and stores Keypoints + Angles """
    if not batch_images: return

    pose_inputs = pose_processor(images=batch_images, boxes=batch_boxes, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        pose_outputs = pose_model(**pose_inputs, dataset_index=torch.tensor([0]).to(DEVICE))

    pose_results = pose_processor.post_process_pose_estimation(pose_outputs, boxes=batch_boxes)

    for i, result in enumerate(pose_results):
        original_idx = batch_indices[i]
        
        if len(result) > 0:
            kpts = result[0]["keypoints"] # Take first person
            
            # 1. Store Raw Keypoints (for drawing later)
            raw_kpts_np = kpts.cpu().numpy()
            
            # 2. Calculate Feature Vector (8 Angles)
            angles = [
                calculate_angle_2d(get_kp(kpts, 6), get_kp(kpts, 8), get_kp(kpts, 10)),  # R_Elbow
                calculate_angle_2d(get_kp(kpts, 5), get_kp(kpts, 7), get_kp(kpts, 9)),   # L_Elbow
                calculate_angle_2d(get_kp(kpts, 8), get_kp(kpts, 6), get_kp(kpts, 12)),  # R_Shoulder
                calculate_angle_2d(get_kp(kpts, 7), get_kp(kpts, 5), get_kp(kpts, 11)),  # L_Shoulder
                calculate_angle_2d(get_kp(kpts, 6), get_kp(kpts, 12), get_kp(kpts, 14)), # R_Hip
                calculate_angle_2d(get_kp(kpts, 5), get_kp(kpts, 11), get_kp(kpts, 13)), # L_Hip
                calculate_angle_2d(get_kp(kpts, 12), get_kp(kpts, 14), get_kp(kpts, 16)),# R_Knee
                calculate_angle_2d(get_kp(kpts, 11), get_kp(kpts, 13), get_kp(kpts, 15)) # L_Knee
            ]
            
            results_container[original_idx] = {
                "exists": True,
                "kpts": raw_kpts_np,
                "features": angles
            }
        else:
            results_container[original_idx] = {"exists": False, "kpts": [], "features": [0]*8}

def process_video_path(video_path, target_frames):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return [], [], 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
    
    frames_img_list = [None] * len(indices)
    results_container = [None] * len(indices)
    
    # Batching
    BATCH_SIZE = 16
    batch_imgs, batch_boxes, batch_idxs = [], [], []

    for i, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret: continue

        # Resize for consistent processing/display
        frame = cv2.resize(frame, (640, 480))
        frames_img_list[i] = frame # Store raw image
        
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # RT-DETR Detection
        det_inputs = det_processor(images=pil_image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            det_outputs = det_model(**det_inputs)
            
        det_results = det_processor.post_process_object_detection(
            det_outputs, target_sizes=torch.tensor([(pil_image.height, pil_image.width)]), threshold=0.3
        )
        
        person_boxes = det_results[0]["boxes"][det_results[0]["labels"] == 0]

        if len(person_boxes) > 0:
            box = person_boxes[0].cpu().numpy()
            # [x, y, w, h] format for VitPose
            box_coco = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
            
            batch_imgs.append(pil_image)
            batch_boxes.append([box_coco])
            batch_idxs.append(i)
        else:
            results_container[i] = {"exists": False, "kpts": [], "features": [0]*8}

        # Process Batch
        if len(batch_imgs) >= BATCH_SIZE:
            process_pose_batch(batch_imgs, batch_boxes, batch_idxs, results_container)
            batch_imgs, batch_boxes, batch_idxs = [], [], []

    # Clean up leftovers
    if batch_imgs:
        process_pose_batch(batch_imgs, batch_boxes, batch_idxs, results_container)
    
    cap.release()
    
    # Pack data cleanly
    processed_data = []
    for i in range(len(indices)):
        if frames_img_list[i] is not None and results_container[i] is not None:
            processed_data.append({
                "image": frames_img_list[i],
                "features": results_container[i]["features"],
                "kpts": results_container[i]["kpts"],
                "exists": results_container[i]["exists"]
            })
            
    return processed_data

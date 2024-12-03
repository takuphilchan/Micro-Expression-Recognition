import os
import cv2
import numpy as np
import mediapipe as mp
import dlib
import mtcnn
from vidstab import VidStab
from scipy.spatial.distance import euclidean

stabilizer_original = VidStab()
stabilizer_augmented = VidStab()

mtcnn_detector = mtcnn.MTCNN()

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.8)

# output_folder = right_mirror  # New folder to save output videos
desired_frame_count = 9
output_resolution = (420, 420)  # Desired output resolution

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Head outline indices
head_outline_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
                        152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# Eye landmarks for alignment
left_eye_indices = [33, 160, 158, 133, 153, 144]
right_eye_indices = [263, 387, 385, 362, 380, 373]
nose_indices = [151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94,2, 164, 0, 11, 12, 13, 14, 15, 16, 17, 18, 200, 199,175,  152]

# Function to parse frame numbers from folder name
def parse_frame_numbers(folder_name):
    parts = folder_name.split('_')
    onset = int(parts[0])
    offset = int(parts[1]) if len(parts) == 2 else int(parts[2])
    apex = (onset + offset) // 2 if len(parts) == 2 else int(parts[1])
    return onset, apex, offset

# Function to select frames using window method and sparse sampling
# window_size=5, sparse_rate=10
# window_size=7, sparse_rate=5
def dynamic_select_frames(onset, apex, offset, window_size, sparse_rate):
    selected_frames = set()
    
    # Total frames is the range from onset to offset
    # total_frames = offset - onset + 1
    
    # Add frames within the window around onset
    for i in range(onset, min(onset + window_size + 1, offset + 1)):
        selected_frames.add(i)
    
    # Add frames within the window around apex
    for i in range(max(apex - window_size, onset), min(apex + window_size + 1, offset + 1)):
        selected_frames.add(i)
    
    # Add frames within the window around offset
    for i in range(max(offset - window_size, onset), offset + 1):
        selected_frames.add(i)
    
    # Sparse sampling outside the windows
    for i in range(onset, offset + 1):
        if i not in selected_frames and (i - onset) % sparse_rate == 0:
            selected_frames.add(i)
    
    # Sort the selected frames
    selected_frames = sorted(selected_frames)
    
    return selected_frames

# return selected_frames
def select_frames(onset, apex, offset):
    """
    Select all frames from onset to offset, inclusive.

    Args:
    onset (int): The onset frame number.
    apex (int): The apex frame number.
    offset (int): The offset frame number.

    Returns:
    list: A list of frame numbers from onset to offset.
    """
    # Ensure all frames from onset to offset are included
    frames = list(range(onset, offset + 1))
    
    return frames
# # Function to get face landmarks using MediaPipe
def get_face_landmarks(image):
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        return results.multi_face_landmarks[0]
    return None

# # Function to get head bounding box from landmarks
def get_head_bounding_box(landmarks, image_shape):
    h, w, _ = image_shape
    x_min, x_max = w, 0
    y_min, y_max = h, 0
    for idx in head_outline_indices:
        x, y = int(landmarks.landmark[idx].x * w), int(landmarks.landmark[idx].y * h)
        if x < x_min: x_min = x
        if x > x_max: x_max = x
        if y < y_min: y_min = y
        if y > y_max: y_max = y
    return x_min, y_min, x_max, y_max

def add_padding_to_bounding_box(bbox, padding, image_shape):
    height, width, _ = image_shape
    if isinstance(bbox, list) or isinstance(bbox, tuple):
        x, y, w, h = bbox
    else:
        raise TypeError("Expected bbox to be a list or tuple with four integers")
    x_new = max(0, x - padding)
    y_new = max(0, y - padding)
    w_new = min(width, x + w + padding) - x_new
    h_new = min(height, y + h + padding) - y_new
    return [x_new, y_new, w_new, h_new]

def crop_to_bounding_box(image, bbox):
    if isinstance(bbox, list) or isinstance(bbox, tuple):
        x, y, w, h = bbox
    else:
        raise TypeError("Expected bbox to be a list or tuple with four integers")
    return image[y:y+h, x:x+w]

def smooth_bounding_boxes(bboxes, smoothing_factor=0.8):
    smoothed_bboxes = []
    prev_bbox = bboxes[0]
    for bbox in bboxes:
        smoothed_bbox = [int(smoothing_factor * prev + (1 - smoothing_factor) * curr) for prev, curr in zip(prev_bbox, bbox)]
        smoothed_bboxes.append(smoothed_bbox)
        prev_bbox = smoothed_bbox
    return smoothed_bboxes

# # Function to get eye centers for alignment
def get_eye_centers(landmarks, image_shape):
    h, w, _ = image_shape
    left_eye = np.mean([(landmarks.landmark[idx].x * w, landmarks.landmark[idx].y * h) for idx in left_eye_indices], axis=0)
    right_eye = np.mean([(landmarks.landmark[idx].x * w, landmarks.landmark[idx].y * h) for idx in right_eye_indices], axis=0)
    return left_eye, right_eye

def get_face_center(landmarks, frame_shape):
    x1 = int(landmarks.landmark[151].x * frame_shape[1])
    y1 = int(landmarks.landmark[151].y * frame_shape[0])
    x2 = int(landmarks.landmark[152].x * frame_shape[1])
    y2 = int(landmarks.landmark[152].y * frame_shape[0])
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    return (center_x, center_y)

# # Function to rotate image to align eyes horizontally
def align_face(image, left_eye, right_eye):
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))
    center = (int((left_eye[0] + right_eye[0]) / 2), int((left_eye[1] + right_eye[1]) / 2))
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)
    new_left_eye = np.dot(M, np.array([left_eye[0], left_eye[1], 1]))
    new_right_eye = np.dot(M, np.array([right_eye[0], right_eye[1], 1]))
    return aligned_image, new_left_eye, new_right_eye, M

def smooth_landmarks(current_landmarks, prev_landmarks, alpha):
    if prev_landmarks is None:
        return current_landmarks
    else:
        smoothed = []
        for curr, prev in zip(current_landmarks, prev_landmarks):
            smoothed_x = int(curr[0] * alpha + prev[0] * (1 - alpha))
            smoothed_y = int(curr[1] * alpha + prev[1] * (1 - alpha))
            smoothed.append((smoothed_x, smoothed_y))
        return smoothed

# Function to get midpoint between two landmarks
def get_midpoint(landmark1, landmark2):
    return ((landmark1[0] + landmark2[0]) // 2, (landmark1[1] + landmark2[1]) // 2)

def compute_slope(landmark1, landmark2):
    if landmark2[0] - landmark1[0] != 0:
        slope = (landmark2[1] - landmark1[1]) / (landmark2[0] - landmark1[0])
    else:
        slope = None  # Handle vertical line case
    
    return slope

def mirror_left(frame, midpoint, slope, nose_width, translation_x, blur_width, rotation_angle=0):
    h, w = frame.shape[:2]
    mirrored_frame = frame.copy()

    # Calculate rotation matrix for alignment
    M_rotate = cv2.getRotationMatrix2D(tuple(midpoint), rotation_angle, 1.0)

    # Calculate mirrored coordinates
    X, Y = np.meshgrid(np.arange(midpoint[0], w), np.arange(h))
    
    if slope is not None:
        intercept = midpoint[1] - slope * midpoint[0]
        D = (X + (Y - intercept) * slope) / (1 + slope**2)
        mirror_X = (2 * D - X - nose_width).astype(int)
        mirror_Y = (2 * D * slope - Y + 2 * intercept).astype(int)
    else:
        mirror_X = (2 * midpoint[0] - X - nose_width).astype(int)
        mirror_Y = Y
    
    mirror_X += translation_x

    # Create mask to handle out-of-bound indices
    valid_mask = (mirror_X >= 0) & (mirror_X < w) & (mirror_Y >= 0) & (mirror_Y < h)

    # Mirror the frame using cv2.warpAffine
    mirrored_frame = cv2.warpAffine(frame, M_rotate, (w, h))
    mirrored_frame[Y[valid_mask], X[valid_mask]] = frame[mirror_Y[valid_mask], mirror_X[valid_mask]]

    # Apply controlled Gaussian blur along the mirror line
    mirrored_frame = apply_blur_near_line(mirrored_frame, midpoint, blur_width)

    return mirrored_frame

def mirror_right(frame, midpoint, slope, nose_width, translation_x, blur_width, rotation_angle=0):
    h, w = frame.shape[:2]
    mirrored_frame = frame.copy()

    # Calculate rotation matrix for alignment
    M_rotate = cv2.getRotationMatrix2D(tuple(midpoint), rotation_angle, 1.0)

    # Calculate mirrored coordinates
    X, Y = np.meshgrid(np.arange(midpoint[0]), np.arange(h))
    
    if slope is not None:
        intercept = midpoint[1] - slope * midpoint[0]
        D = (X + (Y - intercept) * slope) / (1 + slope**2)
        mirror_X = (2 * D - X + nose_width).astype(int)
        mirror_Y = (2 * D * slope - Y + 2 * intercept).astype(int)
    else:
        mirror_X = (2 * midpoint[0] - X + nose_width).astype(int)
        mirror_Y = Y
    
    mirror_X += translation_x

    # Create mask to handle out-of-bound indices
    valid_mask = (mirror_X >= 0) & (mirror_X < w) & (mirror_Y >= 0) & (mirror_Y < h)

    # Mirror the frame using cv2.warpAffine
    mirrored_frame = cv2.warpAffine(frame, M_rotate, (w, h))
    mirrored_frame[Y[valid_mask], X[valid_mask]] = frame[mirror_Y[valid_mask], mirror_X[valid_mask]]

    # Apply controlled Gaussian blur along the mirror line
    mirrored_frame = apply_blur_near_line(mirrored_frame, midpoint, blur_width)

    return mirrored_frame

def apply_blur_near_line(mirrored_frame, midpoint, blur_width):
    h, w = mirrored_frame.shape[:2]
    kernel_size = (15, 15)  # Larger kernel for stronger blur effect
    blurred_frame = cv2.GaussianBlur(mirrored_frame, kernel_size, 0)
    
    # Create mask for blending
    mask = np.zeros((h, w), dtype=np.float32)
    
    # Define regions to apply blur (left and right of midpoint)
    left_bound = max(midpoint[0] - blur_width, 0)
    right_bound = min(midpoint[0] + blur_width, w)
    
    mask[:, left_bound:right_bound] = 1  # Apply full blur within the width bounds
    
    mask = cv2.merge([mask] * 3)  # Create a 3-channel mask
    
    # Blend the original and blurred images using the mask
    blended_frame = (mirrored_frame.astype(np.float32) * (1 - mask) + blurred_frame.astype(np.float32) * mask).astype(np.uint8)
    
    return blended_frame

# Assuming you have defined other necessary functions such as get_face_landmarks, etc.
def decalcomanie_augmentation(frame, mirror_side, landmarks, prev_landmarks, alpha, blur_width=5, slope=None):
    h, w = frame.shape[:2]
    
    # Extract stable landmark indices for use
    stable_indices = [
        151, 152, 358, 129, 4, 5, 6, 197, 195, 9, 8, 168, 164, 0, 11, 12, 13, 14, 15, 16, 1,
        61, 185, 40, 37, 39, 267, 269, 270, 409, 317, 402, 318, 324, 78, 95, 88, 178, 87,
        291, 146, 91, 181, 84, 17, 314, 405, 321, 375, 50, 51, 52, 53, 54, 55, 56, 57, 58,
        59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 79, 80, 81,
        82, 83, 85, 86, 89, 90, 92, 93, 94, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105,
        106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122,
        123, 124, 125, 126, 127, 128, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140,
        141, 142, 143
    ]

    # Extract current landmarks directly without smoothing
    current_landmarks = [(int(landmarks.landmark[idx].x * w), int(landmarks.landmark[idx].y * h)) for idx in stable_indices]

    smoothed_landmarks = smooth_landmarks(current_landmarks, prev_landmarks, alpha)

    # Calculate nose width (assuming it's constant across all frames)
    nose_width = 0  # Placeholder, replace with your actual nose width calculation

    # Calculate midpoint and other necessary parameters
    landmark_129 = smoothed_landmarks[stable_indices.index(129)]
    landmark_358 =  smoothed_landmarks[stable_indices.index(358)]
    midpoint = get_midpoint(landmark_129, landmark_358)
    translation_x = 0  # Adjust as needed based on your requirements

    # Calculate slope between landmarks 151 and 152 if not provided
    if slope is None:
        landmark_151 = current_landmarks[stable_indices.index(151)]
        landmark_152 = current_landmarks[stable_indices.index(152)]
        slope = compute_slope(landmark_151, landmark_152)

    # Choose mirror function based on mirror side
    if mirror_side == 'left':
        mirrored_frame = mirror_left(frame, midpoint, slope, nose_width, translation_x, blur_width)
    elif mirror_side == 'right':
        mirrored_frame = mirror_right(frame, midpoint, slope, nose_width, translation_x, blur_width)
    else:
        mirrored_frame = frame.copy()

    return mirrored_frame, smoothed_landmarks, slope

def calculate_yaw_rotation(prev_landmarks, current_landmarks):
    if len(prev_landmarks) != len(current_landmarks):
        raise ValueError("Landmark sets must have the same length")

    # Indices of relevant landmarks in MediaPipe FaceMesh
    left_eye_indices = [33, 246, 161]
    right_eye_indices = [362, 392, 374]

    # Extract landmark coordinates for left eye
    prev_left_eye = np.mean([prev_landmarks[idx] for idx in left_eye_indices], axis=0)
    curr_left_eye = np.mean([current_landmarks[idx] for idx in left_eye_indices], axis=0)

    # Extract landmark coordinates for right eye
    prev_right_eye = np.mean([prev_landmarks[idx] for idx in right_eye_indices], axis=0)
    curr_right_eye = np.mean([current_landmarks[idx] for idx in right_eye_indices], axis=0)

    # Calculate vectors from landmarks
    prev_eye_vector = prev_right_eye - prev_left_eye
    curr_eye_vector = curr_right_eye - curr_left_eye

    # Normalize vectors
    prev_eye_vector_norm = prev_eye_vector / np.linalg.norm(prev_eye_vector)
    curr_eye_vector_norm = curr_eye_vector / np.linalg.norm(curr_eye_vector)

    # Calculate dot product to find the angle
    dot_product = np.dot(prev_eye_vector_norm, curr_eye_vector_norm)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

    # Convert angle to degrees
    angle_deg = np.degrees(angle)

    return angle_deg

# Function to construct video from selected frames
def construct_video(video_folder_path, frames_to_use, output_path_original, output_path_augmented, mirror_side, movement_threshold=10, rotation_threshold=10):
    video_frames_original = []
    video_frames_augmented = []
    bboxes = []
    bboxes_aug = []
    prev_landmarks = None
    prev_landmarks_aug = None
    alpha = 0.0  # Smoothing factor
    ema_bbox = None
    frame_count = len(frames_to_use)
    # print(f"Total frames: {frame_count}")
    
    # Initialize slope and intercept as None
    initial_slope = None
    initial_intercept = None

    landmark_pairs = [(129, 358)]
    initial_distances = {}
    initial_nose_width = None
    distance_threshold = 0.03

    significant_movement_detected = False

    for frame_number in frames_to_use:
        frame_path = os.path.join(video_folder_path, f"{frame_number}.png")
        if not os.path.exists(frame_path):
            frame_path = os.path.join(video_folder_path, f"img{frame_number}.jpg")
        
        frame = cv2.imread(frame_path)
        if frame is None:
            break
        
        landmarks = get_face_landmarks(frame)
        if landmarks:
            current_landmarks = [(landmarks.landmark[idx].x, landmarks.landmark[idx].y) for idx in range(468)]
            if prev_landmarks:
                # Calculate the movement between frames
                movement = sum(euclidean(curr, prev) for curr, prev in zip(current_landmarks, prev_landmarks)) / len(current_landmarks)
                if movement > movement_threshold:
                    significant_movement_detected = True
                    break
                # print(f"Movement: {movement}")
                # print(f"Movement threshold: {movement_threshold}")
                # Check rotation about y-axis (yaw)
                rotation_angle = calculate_yaw_rotation(prev_landmarks, current_landmarks)
                # print(f"Rotation angle: {rotation_angle}")
                # print(f"Movement: {movement}")
                if abs(rotation_angle) > rotation_threshold:
                    significant_movement_detected = True
                    break
            prev_landmarks = current_landmarks

            bbox = get_head_bounding_box(landmarks, frame.shape)
            bboxes.append(bbox)
            left_eye, right_eye = get_eye_centers(landmarks, frame.shape)
            aligned_frame, new_left_eye, new_right_eye, M_orig = align_face(frame, left_eye, right_eye)
        else:
            aligned_frame = frame.copy()
        
        video_frames_original.append((aligned_frame, landmarks, new_left_eye, new_right_eye))

    if significant_movement_detected and M_orig is not None:
        # print("Significant head movement detected. Skipping augmentation.")
        process_and_save_video(video_frames_original, bboxes, output_path_original, M_orig)
    else:
        # print("Negligible head movement detected. Performing augmentation.")
        for frame_number in frames_to_use:
            frame_path = os.path.join(video_folder_path, f"{frame_number}.png")
            if not os.path.exists(frame_path):
                frame_path = os.path.join(video_folder_path, f"img{frame_number}.jpg")
            
            frame = cv2.imread(frame_path)
            if frame is None:
                break
            
            landmarks = get_face_landmarks(frame)

            # Call the augmentation function
            augmented_frame, prev_landmarks_aug, initial_slope = decalcomanie_augmentation(
                frame, mirror_side, landmarks, prev_landmarks_aug, alpha, blur_width=5, 
                slope=initial_slope
            )
            
            landmarks_aug = get_face_landmarks(augmented_frame)
            if landmarks_aug and len(landmarks_aug.landmark) > max(129, 358):
                bbox_aug = get_head_bounding_box(landmarks_aug, augmented_frame.shape)
                bboxes_aug.append(bbox_aug)
                left_eye_aug, right_eye_aug = get_eye_centers(landmarks_aug, augmented_frame.shape)
                aligned_augmented_frame, new_left_eye_aug, new_right_eye_aug, M_aug = align_face(augmented_frame, left_eye_aug, right_eye_aug)

                # Calculate distances for all landmark pairs on the augmented frame
                for pair in landmark_pairs:
                    idx1, idx2 = pair
                    distance = euclidean((landmarks_aug.landmark[idx1].x, landmarks_aug.landmark[idx1].y),
                                         (landmarks_aug.landmark[idx2].x, landmarks_aug.landmark[idx2].y))
                    
                    if pair not in initial_distances:
                        initial_distances[pair] = distance
                    elif abs((distance - initial_distances[pair]) / initial_distances[pair]) > distance_threshold:
                        # print(f"Significant change detected for landmark pair {pair} on augmented frame. Skipping saving the augmented video.")
                        process_and_save_video(video_frames_original, bboxes, output_path_original, M_orig)
                        return
                    # print(abs((distance - initial_distances[pair]) / initial_distances[pair]),  distance_threshold)
            else:
                aligned_augmented_frame = augmented_frame.copy()
            
            video_frames_augmented.append((aligned_augmented_frame, landmarks_aug, new_left_eye_aug, new_right_eye_aug))

        process_and_save_video(video_frames_original, bboxes, output_path_original, M_orig)
        process_and_save_video(video_frames_augmented, bboxes_aug, output_path_augmented, M_aug)

def process_and_save_video(video_frames, bboxes, output_path, M):
    if not video_frames:
        # print(f"No frames found for {output_path}")
        return False

    smoothed_bboxes = smooth_bounding_boxes(bboxes)
    cropped_frames = []
    for idx, (aligned_frame, landmarks, new_left_eye, new_right_eye) in enumerate(video_frames):
        if landmarks:
            x_min, y_min, x_max, y_max = smoothed_bboxes[idx]
            new_landmarks = []
            for landmark_idx in head_outline_indices:
                point = np.array([landmarks.landmark[landmark_idx].x * aligned_frame.shape[1],
                                  landmarks.landmark[landmark_idx].y * aligned_frame.shape[0], 1.0])
                new_point = np.dot(M, point)
                new_landmarks.append((int(new_point[0]), int(new_point[1])))

            mask = np.zeros(aligned_frame.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(new_landmarks, dtype=np.int32)], (255, 255, 255))
            cropped_frame = cv2.bitwise_and(aligned_frame, mask)
            black_background = np.zeros_like(aligned_frame)  # Changed to create a black background
            mask_inv = cv2.bitwise_not(mask)
            black_background = cv2.bitwise_and(black_background, mask_inv)
            final_frame = cv2.add(cropped_frame, black_background)
            cropped_frames.append(final_frame[y_min:y_max, x_min:x_max])

    resized_frames = [cv2.resize(frame, output_resolution) for frame in cropped_frames]

    frame_height, frame_width, _ = resized_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out = cv2.VideoWriter(output_path, fourcc, 25.0, (frame_width, frame_height))
    for frame in resized_frames:
        out.write(frame)
    out.release()
    # print(f"Video saved to {output_path}")
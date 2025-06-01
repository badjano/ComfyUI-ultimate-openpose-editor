import math
import json
import numpy as np
import matplotlib
import cv2
from comfy.utils import ProgressBar

eps = 0.01

def scale(point, scale_factor, pivot):
    return [(point[0] - pivot[0]) * scale_factor + pivot[0], (point[1] - pivot[1]) * scale_factor + pivot[1]]

def draw_pose_json(pose_json, resolution_x, show_body, show_face, show_hands, pose_marker_size, face_marker_size, hand_marker_size, hands_scale, body_scale, head_scale, overall_scale):
    pose_imgs = []
    
    if pose_json:
        if pose_json.startswith('{'):
            pose_json = '[{}]'.format(pose_json)
        images = json.loads(pose_json)
        pbar = ProgressBar(len(images))
        for image in images:
            if 'people' not in image:
                pbar.update(len(images))
                return pose_imgs
            figures = image['people']
            H = image['canvas_height']
            W = image['canvas_width']
            
            bodies = []
            candidate = []
            subset = [[]]
            faces = []
            hands = []
            pivot = [W * 0.5, H * 0.5]
            
            for figure_idx, figure in enumerate(figures):
                body = []
                face = []
                lhand = []
                rhand = []
                if 'pose_keypoints_2d' in figure:
                    body = figure['pose_keypoints_2d']
                if 'face_keypoints_2d' in figure:
                    face = figure['face_keypoints_2d']
                if 'hand_left_keypoints_2d' in figure:
                    lhand = figure['hand_left_keypoints_2d']
                if 'hand_right_keypoints_2d' in figure:
                    rhand = figure['hand_right_keypoints_2d']
                    
                face_offset = [0, 0]
                lhand_offset = [0, 0]
                rhand_offset = [0, 0]

                lhand_pivot = [W * 0.25, H * 0.5]
                rhand_pivot = [W * 0.75, H * 0.5]
                face_pivot = [W * 0.5, H * 0.5]

                if body:
                    candidate_start_idx = len(candidate)
                    
                    index = 0
                    for i in range(0,len(body),3):
                        p = body[i:i+2]
                        confidence = body[i+2]
                        
                        if body_scale != 1.0:
                            point = scale(p, body_scale, pivot)
                        else:
                            point = p[:]
                            
                        candidate.append(point)
                        index += 1
                    
                    figure_head_idx = candidate_start_idx
                    if figure_head_idx < len(candidate):
                        face_offset = [candidate[figure_head_idx][0] - body[0], candidate[figure_head_idx][1] - body[1]]
                        face_offset = [a*0.8 for a in face_offset]
                    else:
                        face_offset = [0, 0]

                    wrist_left_idx = candidate_start_idx + 7
                    wrist_right_idx = candidate_start_idx + 4
                    
                    if wrist_left_idx < len(candidate) and len(body) > 22:
                        lhand_offset = [candidate[wrist_left_idx][0] - body[21], candidate[wrist_left_idx][1] - body[22]]
                        lhand_pivot = candidate[wrist_left_idx]
                    else:
                        lhand_offset = [0, 0]
                        
                    if wrist_right_idx < len(candidate) and len(body) > 13:
                        rhand_offset = [candidate[wrist_right_idx][0] - body[12], candidate[wrist_right_idx][1] - body[13]]
                        rhand_pivot = candidate[wrist_right_idx]
                    else:
                        rhand_offset = [0, 0]

                    if figure_head_idx < len(candidate):
                        face_pivot = candidate[figure_head_idx]

                    if not subset[0]:
                        subset[0].extend([candidate_start_idx+(i//3) if body[i+2]>0 else -1 for i in range(0,len(body),3)])
                    else:
                        new_subset = [candidate_start_idx+(i//3) if body[i+2]>0 else -1 for i in range(0,len(body),3)]
                        subset.append(new_subset)

                if face:
                    f = []
                    for i in range(0,len(face),3):
                        p = face[i:i+2]
                        confidence = face[i+2]
                        p_offset = [p[0] + face_offset[0], p[1] + face_offset[1]]
                        p_scaled = scale(p_offset, head_scale, face_pivot)
                        f.append(p_scaled)
                    faces.append(f)
                    
                if lhand:
                    lh = []
                    for i in range(0, len(lhand), 3):
                        p = lhand[i:i+2]
                        p_offset = [p[0] + lhand_offset[0], p[1] + lhand_offset[1]]
                        p_scaled = scale(p_offset, hands_scale, lhand_pivot)
                        lh.append(p_scaled)
                    hands.append(lh)
                    
                if rhand:
                    rh = []
                    for i in range(0, len(rhand), 3):
                        p = rhand[i:i+2]
                        p_offset = [p[0] + rhand_offset[0], p[1] + rhand_offset[1]]
                        p_scaled = scale(p_offset, hands_scale, rhand_pivot)
                        rh.append(p_scaled)
                    hands.append(rh)

            
            if hands:
                hands = [[scale(lm, overall_scale, pivot) for lm in hand] for hand in hands]
            if faces:
                faces = [[scale(lm, overall_scale, pivot) for lm in face] for face in faces]
            if candidate:
                candidate = [scale(lm, overall_scale, pivot) for lm in candidate]

            if candidate:
                candidate = np.array(candidate).astype(float)
                subset = np.array(subset)
                max_x = np.max(candidate[...,0]) if len(candidate) > 0 else 0
                max_y = np.max(candidate[...,1]) if len(candidate) > 0 else 0
                normalized = max(max_x, max_y)
                if normalized > 2.0:
                    candidate[...,0] = np.clip(candidate[...,0] / float(W), 0, 1)
                    candidate[...,1] = np.clip(candidate[...,1] / float(H), 0, 1)
                    
            if faces:
                faces = np.array(faces).astype(float)
                max_x = np.max(faces[...,0]) if len(faces) > 0 else 0
                max_y = np.max(faces[...,1]) if len(faces) > 0 else 0
                normalized = max(max_x, max_y)
                if normalized > 2.0:
                    faces[...,0] = np.clip(faces[...,0] / float(W), 0, 1)
                    faces[...,1] = np.clip(faces[...,1] / float(H), 0, 1)
                    
            if hands:
                hands = np.array(hands).astype(float)
                max_x = np.max(hands[...,0]) if len(hands) > 0 else 0
                max_y = np.max(hands[...,1]) if len(hands) > 0 else 0
                normalized = max(max_x, max_y)
                if normalized > 2.0:
                    hands[...,0] = np.clip(hands[...,0] / float(W), 0, 1)
                    hands[...,1] = np.clip(hands[...,1] / float(H), 0, 1)
                    
            bodies = dict(candidate=candidate, subset=subset)
            pose = dict(bodies=bodies, faces=faces, hands=hands)
            pose = dict(bodies=bodies if show_body else {'candidate':[], 'subset':[]}, faces=faces if show_face else [], hands=hands if show_hands else [])
            
            W_scaled = resolution_x
            if resolution_x < 64:
                W_scaled = W
            H_scaled = int(H*(W_scaled*1.0/W))
            
            pose_img = draw_pose(pose, H_scaled, W_scaled, pose_marker_size, face_marker_size, hand_marker_size)
            pose_imgs.append(pose_img)
            pbar.update(1)

    return pose_imgs

def draw_pose(pose, H, W, pose_marker_size, face_marker_size, hand_marker_size):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    if len(candidate) > 0:
        canvas = draw_bodypose(canvas, candidate, subset, pose_marker_size)

    if len(hands) > 0:
        canvas = draw_handpose(canvas, hands, hand_marker_size)

    if len(faces) > 0:
        canvas = draw_facepose(canvas, faces, face_marker_size)

    return canvas

def draw_bodypose(canvas, candidate, subset, pose_marker_size):
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    # stickwidth = 4

    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            Y = candidate[index.astype(int), 0] * float(W)
            X = candidate[index.astype(int), 1] * float(H)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), pose_marker_size), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, colors[i])

    canvas = (canvas * 0.6).astype(np.uint8)

    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            x = int(x * W)
            y = int(y * H)
            cv2.circle(canvas, (int(x), int(y)), pose_marker_size, colors[i], thickness=-1)

    return canvas


def draw_handpose(canvas, all_hand_peaks, hand_marker_size):
    H, W, C = canvas.shape

    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

    for peaks in all_hand_peaks:
        peaks = np.array(peaks)

        for ie, e in enumerate(edges):
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            x2 = int(x2 * W)
            y2 = int(y2 * H)
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                cv2.line(canvas, (x1, y1), (x2, y2), matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * 255, thickness=1 if hand_marker_size == 0 else hand_marker_size)

        joint_size=0
        if hand_marker_size < 2:
            joint_size = hand_marker_size + 1
        else:
            joint_size = hand_marker_size + 2
        for i, keyponit in enumerate(peaks):
            x, y = keyponit
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), joint_size, (0, 0, 255), thickness=-1)
    return canvas


def draw_facepose(canvas, all_lmks, face_marker_size):
    H, W, C = canvas.shape
    for lmks in all_lmks:
        lmks = np.array(lmks)
        for lmk in lmks:
            x, y = lmk
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), face_marker_size, (255, 255, 255), thickness=-1)
    return canvas

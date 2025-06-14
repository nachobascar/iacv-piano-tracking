import cv2
import numpy as np
import mediapipe as mp
import argparse
import json
import os


# MediaPipe setup
mp_hands = mp.solutions.hands


def load_or_compute_homography(
    folder: str, params: dict
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Piano rectification and key polygons extraction.

    Parameters
    ----------
    folder : str       - name of the input folder with reference.png
    params : dict      - parameters for the rectification, see inputs/<folder>/params.json

    Returns
    -------
    H : np.ndarray     - 3x3 homography matrix to rectify the keyboard
    key_polys : list   - list of polygons for each key in the rectified image
    """

    # Create output folder if it doesn't exist
    output_folder = os.path.join("outputs", folder, "homography")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    def get_output_path(filename: str) -> str:
        return os.path.join(output_folder, filename)

    # 1. Calculate Canny edges of the keyboard from the reference image
    img = cv2.imread(os.path.join("inputs", folder, "reference.png"))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 80)

    cv2.imwrite(get_output_path("edges.png"), edges)

    # 2.  Hough-detect the vertical and horizontal lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 120, minLineLength=80, maxLineGap=20)
    horiz, vert = [], []
    img_lines = img.copy()

    padding = 7  # Padding to avoid image edges

    for x1, y1, x2, y2 in lines[:, 0]:
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < 20 and y1 > padding and y2 > padding:
            if (
                "MIN_REFERENCE_Y" in params
            ):  # Optional parameter to manually set keyboard Y boundaries
                if y1 < params["MIN_REFERENCE_Y"] or y2 < params["MIN_REFERENCE_Y"]:
                    continue
            horiz.append((x1, y1, x2, y2))
            cv2.line(img_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
        elif 90 - abs(angle) < 20 and x1 > padding and x2 > padding:
            vert.append((x1, y1, x2, y2))
            cv2.line(img_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite(get_output_path("lines.png"), img_lines)

    # pick extreme lines → keyboard rectangle
    top = min(horiz, key=lambda l: min(l[1], l[3]))
    bottom = max(horiz, key=lambda l: max(l[1], l[3]))
    left = min(vert, key=lambda l: min(l[0], l[2]))
    right = max(vert, key=lambda l: max(l[0], l[2]))

    def intersect(l1, l2):
        # l = (x1,y1,x2,y2)  homogeneous intersection of two lines
        L1 = np.cross([l1[0], l1[1], 1], [l1[2], l1[3], 1])
        L2 = np.cross([l2[0], l2[1], 1], [l2[2], l2[3], 1])
        p = np.cross(L1, L2)
        p = p / p[2]
        return p[:2]

    tl = intersect(top, left)
    tr = intersect(top, right)
    br = intersect(bottom, right)
    bl = intersect(bottom, left)

    # Show the detected keyboard boundaries
    img_lines = img.copy()
    for l in [top, bottom, left, right]:
        cv2.line(img_lines, (l[0], l[1]), (l[2], l[3]), (255, 0, 0), 2)
    for pt in [tl, tr, br, bl]:
        cv2.circle(img_lines, tuple(pt.astype(int)), 5, (0, 255, 255), -1)
    cv2.imwrite(get_output_path("keyboard_edges.png"), img_lines)

    # 3.  rectify (front-parallel)
    RECT_W = params["RECT_W"]  # width of the rectified image
    RECT_H = params["RECT_H"]  # height of the rectified image
    dst = np.float32([[0, 0], [RECT_W, 0], [RECT_W, RECT_H], [0, RECT_H]])
    H, _ = cv2.findHomography(np.float32([tl, tr, br, bl]), dst)
    rect = cv2.warpPerspective(img, H, (RECT_W, RECT_H))

    # 3.1. Optional: apply black-and-white binarization
    if "REFERENCE_THRESH" in params:
        rect = cv2.threshold(
            cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY),
            params["REFERENCE_THRESH"],
            255,
            cv2.THRESH_BINARY,
        )[1]
        rect = cv2.cvtColor(rect, cv2.COLOR_GRAY2BGR)  # convert back to BGR for saving
    cv2.imwrite(get_output_path("rectified.png"), rect)

    # 4.  find white-key boundaries with Gaussian Blur vertical projection
    proj = rect[:, :, 1].mean(axis=0)  # green channel ≈ brightness
    proj = cv2.GaussianBlur(proj, (1, 9), 0)
    d = np.diff(proj, axis=0)  # vertical gradient

    edges_x = np.where(d > 1)[0]
    # merge close peaks → keys boundaries
    bounds = [edges_x[0]]
    for x in edges_x[1:]:
        if x - bounds[-1] > params["MIN_KEY_SIZE"] or (
            len(bounds) == 1 and x - bounds[-1] > params["MIN_KEY_SIZE_CORNER"]
        ):
            bounds.append(x)
    bounds.append(RECT_W)  # rightmost edge

    # Show bounds found
    img_bounds = rect.copy()
    for x in bounds:
        cv2.line(img_bounds, (x, 0), (x, RECT_H), (255, 0, 0), 1)
    cv2.imwrite(get_output_path("bounds.png"), img_bounds)

    # 5.  build key polygons in rectified space
    key_polys_rect = []
    first_key = params[
        "FIRST_KEY"
    ]  # first key on the polychromatic pattern (black, white)
    white_i = 0
    for i in range(first_key, params["N_KEYS"] + first_key):
        is_black = (i % 12) in {1, 3, 6, 8, 10}
        if is_black:
            # black keys sit 2⁄3 height, two-thirds width of white
            wL = bounds[white_i - 1]
            wR = bounds[white_i]
            xL = wR - 0.3 * (wR - wL)
            xR = wR + 0.3 * (wR - wL)
            key_height = RECT_H * 0.6
            key_polys_rect.append(
                np.float32([[xL, 0], [xR, 0], [xR, key_height], [xL, key_height]])
            )
        else:
            xL, xR = bounds[white_i], bounds[white_i + 1]
            key_polys_rect.append(
                np.float32([[xL, 0], [xR, 0], [xR, RECT_H], [xL, RECT_H]])
            )
            white_i += 1

    # Show key polygons
    img_keys = rect.copy()
    for poly in key_polys_rect:
        cv2.polylines(
            img_keys, [poly.astype(int)], isClosed=True, color=(0, 255, 0), thickness=1
        )
    cv2.imwrite(get_output_path("key_polys.png"), img_keys)

    # 6.  map polygons back to original image coords
    H_inv = np.linalg.inv(H)
    key_polys = []
    img_keys = img.copy()
    for poly in key_polys_rect:
        pts = cv2.perspectiveTransform(poly.reshape(-1, 1, 2), H_inv).reshape(-1, 2)
        key_polys.append(pts)
        cv2.polylines(
            img_keys, [pts.astype(int)], isClosed=True, color=(0, 255, 0), thickness=1
        )
    cv2.imwrite(get_output_path("key_polys_original.png"), img_keys)

    return H, key_polys


def build_skin_mask(bgr: np.ndarray, params: dict) -> np.ndarray:
    """
    Hand/skin mask that survives warm/yellow lighting.

    Parameters
    ----------
    bgr : uint8 HxWx3  - current frame

    Returns
    -------
    mask : uint8 HxW   - 255 where skin, 0 elsewhere
    """

    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    Y, Cr, Cb = cv2.split(ycc)
    H, S, V = cv2.split(hsv)

    # 1. classic YCrCb window (very permissive)
    mask_ycc = (Cr > 133) & (Cr < 185) & (Cb > 77) & (Cb < 127)

    # 2. add saturation test
    mask_sat = S > params.get("SKIN_SATURATION_THRESH", 120)

    # 3. discard very bright pixels (white keys)
    mask_not_white = Y < params.get("SKIN_WHITE_THRESH", 200)

    mask = mask_ycc & mask_sat & mask_not_white
    mask = mask.astype(np.uint8) * 255

    # 4. morphological clean-up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def is_key_toggled(
    folder: str,
    key_poly: np.ndarray,
    ref_img: np.ndarray,
    curr_img: np.ndarray,
    params: dict,
    frac_thresh: float,
    bottom_ratio=1,
    args=None,
) -> float:
    """
    Check if a key is toggled based on how different the reference image and the frame are.
    Parts which are not a key (e.g. finger on key, hand shadows, etc.) must be removed

    Parameters
    ----------
    folder : str       - name of the input folder with reference.png
    key_poly : np.ndarray - polygon of the key in rectified coordinates
    ref_img : np.ndarray - reference image of the keyboard
    curr_img : np.ndarray - current frame image
    params : dict      - parameters for the rectification, see inputs/<folder>/params.json
    frac_thresh : float - threshold for the fraction of changed pixels
    bottom_ratio : float - ratio of the key height to consider (to avoid fingertip occlusion)
    args : argparse.Namespace - command line arguments

    Returns
    -------
    score : float      - fraction of changed pixels in the key area
    """

    output_folder = os.path.join("outputs", folder, "is_toggled")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    def get_output_path(filename: str) -> str:
        return os.path.join(output_folder, filename)

    mask_key = np.zeros(curr_img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask_key, [key_poly.astype(np.int32)], 255)
    # keep only bottom X %  ––  avoids fingertip occlusion
    y_min = np.min(key_poly[:, 1])
    y_max = np.max(key_poly[:, 1])
    y_cut = y_max - (y_max - y_min) * bottom_ratio  # 1 = full key height. Default 1
    mask_bottom = np.zeros_like(mask_key)
    mask_bottom[int(y_cut) :, :] = 255
    mask_roi = cv2.bitwise_and(mask_key, mask_bottom)

    # Show the key mask
    if args.show:
        cv2.imwrite(get_output_path("mask_key.png"), mask_roi)

    # 2.  Skin mask
    mask_skin = build_skin_mask(curr_img, params)

    # Show the image with the mask colored in red
    if args.show:
        img_with_mask = curr_img.copy()
        img_with_mask[mask_skin == 255] = [0, 0, 255]  # Red for skin
        cv2.imwrite(get_output_path("mask_skin.png"), img_with_mask)

    # ROI pixels not covered by hands
    visible = cv2.bitwise_and(mask_roi, cv2.bitwise_not(mask_skin))
    n_visible = cv2.countNonZero(visible)
    if n_visible < 30:  # almost fully occluded --> can’t decide
        return False, 0.0

    # 3. Pixel-difference test
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.threshold(ref_gray, 70, 255, cv2.THRESH_BINARY)[1]
    frame_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.threshold(frame_gray, 70, 255, cv2.THRESH_BINARY)[1]

    # Show gray images
    if args.show:
        cv2.imwrite(get_output_path("ref_gray.png"), ref_gray)
        cv2.imwrite(get_output_path("frame_gray.png"), frame_gray)

    # Check if the images are loaded correctly
    if ref_gray is None:
        raise ValueError("ref_gray is None. Check the image loading process.")

    if visible is None:
        raise ValueError("Mask 'visible' is None. Check how it's created.")

    if ref_gray.shape[:2] != visible.shape[:2]:
        raise ValueError(
            f"Shape mismatch: ref_gray shape {ref_gray.shape} vs visible shape {visible.shape}"
        )
    if visible.dtype != np.uint8:
        visible = visible.astype(np.uint8)

    # Show masked images
    if args.show:
        ref_masked = cv2.bitwise_and(ref_gray, ref_gray, mask=visible)
        cv2.imwrite(get_output_path("ref_masked.png"), ref_masked)
        frame_masked = cv2.bitwise_and(frame_gray, frame_gray, mask=visible)
        cv2.imwrite(get_output_path("frame_masked.png"), frame_masked)

    # Get absolute difference
    diff = cv2.absdiff(frame_gray, ref_gray)
    # Remove pixels that are not in the key area
    changed = cv2.bitwise_and(diff, diff, mask=visible)
    # Threshold the difference to get a binary mask of changed pixels
    changed_mask = (changed > 20).astype(np.uint8) * 255

    if args.show:
        cv2.imwrite(get_output_path("changed.png"), changed_mask)

    n_changed = cv2.countNonZero(changed_mask)

    # Score is the fraction of changed pixels in the visible area
    score = n_changed / n_visible
    pressed = score >= frac_thresh
    return score


def get_keys_being_touched(
    folder: str, img: np.ndarray, key_polys: list, args=None
) -> tuple[list, np.ndarray]:
    """
    Detect which keys are being touched in the current frame.

    Parameters
    ----------
    folder : str       - name of the input folder with reference.png
    img : np.ndarray   - current frame image
    key_polys : list   - list of polygons for each key in the rectified image
    args : argparse.Namespace - command line arguments

    Returns
    -------
    touched_keys : list - list of indices of touched keys
    annotated_image : np.ndarray - image with drawn hand landmarks and touched keys
    """
    output_folder = os.path.join("outputs", folder, "keys_touched")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    def get_output_path(filename):
        return os.path.join(output_folder, filename)

    # 1. Get MediaPipe hand landmarks
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=3,
        min_detection_confidence=0.4,
    ) as hands:
        custom_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(custom_img)

        if args.show:
            cv2.imwrite(get_output_path("input_frame.png"), img)

    annotated_image = img.copy()
    if not results.multi_hand_landmarks:
        print("No hands detected in the frame.")
        return [], img

    hands_landmarks = results.multi_hand_landmarks
    if args.show:
        print(f"Detected {len(hands_landmarks)} hands in the frame.")

    fingertips = []
    for hand_landmarks in hands_landmarks:
        if args.show:
            mp.solutions.drawing_utils.draw_landmarks(
                annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )
        # Extract fingertip positions
        fingertips.extend(
            [
                hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
                hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP],
            ]
        )
    if args.show:
        cv2.imwrite(get_output_path("hand_landmarks.png"), annotated_image)

    # 2. Project fingertips into rectified coordinates
    fingertips_rect = []
    for fingertip in fingertips:
        x = int(fingertip.x * img.shape[1])
        y = int(fingertip.y * img.shape[0])
        z = fingertip.z
        fingertips_rect.append((x, y, z))
    if args.show:
        for x, y, z in fingertips_rect:
            cv2.circle(annotated_image, (x, y), 5, (0, 255, 255), -1)
        cv2.imwrite(get_output_path("fingertips.png"), annotated_image)

    # 3. Check which keys are touched
    touched_keys = []
    for i, poly in enumerate(key_polys):
        poly = poly.astype(np.int32)
        for x, y, z in fingertips_rect:
            if cv2.pointPolygonTest(poly, (x, y), False) >= 0:
                if z < 0.02:
                    touched_keys.append(i)
                    if args.show:
                        cv2.polylines(
                            annotated_image,
                            [poly],
                            isClosed=True,
                            color=(0, 0, 255),
                            thickness=2,
                        )
                        cv2.putText(
                            annotated_image,
                            f"Key {i}",
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            1,
                        )
    if args.show:
        cv2.imwrite(get_output_path("touched_keys.png"), annotated_image)

    return touched_keys, annotated_image


def process_video(
    args: argparse.Namespace, cap: cv2.VideoCapture, params: dict
) -> dict:
    """
    Process the video to detect pressed keys.

    Parameters
    ----------
    args : argparse.Namespace - command line arguments
    cap : cv2.VideoCapture - video capture object
    params : dict - parameters for the rectification, see inputs/<folder>/params.json

    Returns
    -------
    final_results : dict - dictionary with frame numbers as keys and lists of pressed keys as values
    """

    # 1. Compute key polygons and homography matrix
    H, key_polys = load_or_compute_homography(args.input, params)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise ValueError("Invalid FPS value. Please check the video file.")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        raise ValueError("Video file has no frames.")

    print(f"Processing video: {args.input} at {fps} FPS, {frame_count} frames")

    ref_img = cv2.imread(os.path.join("inputs", args.input, "reference.png"))

    output_folder = os.path.join("outputs", args.input, "frames")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get first frame: allows to start from a specific frame (if the video has an intro)
    start_frame = params["FIRST_FRAME"]
    frame = start_frame
    cont = 0

    final_results = {}

    # 2. Process video frames
    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, first_frame = cap.read()
        if not ret:
            cap.release()
            break

        # 2.1. Get keys being touched
        possible_keys, img_fingers = get_keys_being_touched(
            args.input, first_frame, key_polys, args=args
        )

        # 2.2. Check if the keys are toggled
        pressed_keys = []
        for i in possible_keys:
            poly = key_polys[i]
            # 2.2.1. Check if the key is toggled
            score = is_key_toggled(
                args.input,
                poly,
                ref_img,
                first_frame,
                params,
                frac_thresh=params["FRAC_THRESH"],
                args=args,
            )

            if score > params["FRAC_THRESH"]:
                pressed_keys.append(i)
                if args.show:
                    # Put green border around the key
                    cv2.polylines(
                        img_fingers,
                        [poly.astype(np.int32)],
                        isClosed=True,
                        color=(0, 255, 0),
                        thickness=2,
                    )
                    print(f"Frame {frame}: Key {i} is toggled with score {score:.2f}")

        if args.show:
            cv2.imwrite(os.path.join(output_folder, f"frame_{frame}.png"), img_fingers)
            print("Touched Keys:", possible_keys)
            print("Pressed Keys:", pressed_keys)

        final_results[frame] = pressed_keys
        frame += params["FRAME_STEP"]
        cont += 1

        MAX_ITERS = 250  # Maximum number of iterations to process

        if cont >= MAX_ITERS:  # or frame > 6 * 60:
            break
        print(
            f"Processed frame {frame} / {start_frame + MAX_ITERS * params['FRAME_STEP']}"
        )

    if args.show:
        for frame, keys in final_results.items():
            print(f"Frame {frame}: Pressed Keys: {keys}")

    return final_results


def soften_results(results: dict) -> dict:
    """
    Softens the results by ensuring each key is pressed for at least min_frames
    and merging frames from 2 past and future.

    Parameters
    ----------
    results : dict - dictionary with frame numbers as keys and lists of pressed keys as values

    Returns
    -------
    results : dict - softened results with frame numbers as keys and lists of unique pressed keys as values
    """

    for frame, keys in results.items():
        results[frame] = list(set(keys))  # Remove duplicates

    return results

    """
        Softening was shown to missperform in some cases, so it is disabled.
    """

    frame_step = params["FRAME_STEP"]
    # Softening: ensure each key is pressed for at least min_frames
    for frame, keys in results.items():
        new_keys = keys.copy()
        for key in keys:
            int_frame = int(frame)
            frames_to_look = [
                str(f)
                for f in [
                    int_frame - 2 * frame_step,
                    int_frame - frame_step,
                    int_frame + frame_step,
                    int_frame + 2 * frame_step,
                ]
            ]
            count = 0
            for f in frames_to_look:
                if f in results and key in results[f]:
                    count += 1
            if count < 0:
                # If no other frame has this key, remove it
                new_keys.remove(key)
        results[frame] = new_keys

    # Merge frames from 2 past and future
    merged_results = {}
    for frame, keys in results.items():
        int_frame = int(frame)
        to_merge = [
            str(k)
            for k in range(int_frame - 0 * frame_step, int_frame + 1 * frame_step)
        ]
        merged_keys = set()
        for f in to_merge:
            if f in results:
                merged_keys.update(results[f])
        merged_results[frame] = list(merged_keys)

    return merged_results


# Create MIDI file from results and fps
def create_midi_from_results(results: dict, fps: float, output_file: str, params: dict):
    """
    Create a MIDI file from the results of the key presses.

    Parameters
    ----------
    results : dict - dictionary with frame numbers as keys and lists of pressed keys as values
    fps : float - frames per second of the video
    output_file : str - path to save the MIDI file
    params : dict - parameters for the rectification, see inputs/<folder>/params.json
    """

    from mido import MidiFile, MidiTrack, Message, MetaMessage

    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    track.append(MetaMessage("key_signature", key="C", time=0))
    track.append(MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    track.append(
        MetaMessage(
            "set_tempo",
            tempo=int(100_000_000 / 20 * params["TEMPO_ADJUST"] / fps),
            time=0,
        )
    )

    active_keys = set()
    frames = []
    for frame, keys in results.items():
        frames.append([int(frame), keys])
    frames.sort(key=lambda x: x[0])  # Sort by frame number

    start_frame = frames[0][0] if frames else 0

    last_frame_time = start_frame

    for frame, keys in frames:
        frame = int(frame)  # Ensure frame is an integer
        time = int((frame / fps))  # Convert to milliseconds

        new_active_keys = set()  # Copy current active keys

        frame_time = frame - last_frame_time

        KEY_OFFSET = params["FIRST_KEY"] + 12 * params["OCTAVE_OFFSET"]

        for key in set(keys):
            if key not in active_keys:
                print(f"Key {key} pressed at frame {frame}, time {time} s")
                note_on = Message(
                    "note_on",
                    note=key + KEY_OFFSET,
                    velocity=64,
                    time=frame_time * 20,
                )
                frame_time = 0
                last_frame_time = frame
                track.append(note_on)
                active_keys.add(key)
            new_active_keys.add(key)
        for key in active_keys - new_active_keys:
            # print(f"Key {key} released at frame {frame}, time {time} s")
            note_off = Message(
                "note_off", note=key + KEY_OFFSET, velocity=64, time=frame_time * 20
            )
            frame_time = 0
            last_frame_time = frame
            track.append(note_off)
            active_keys.remove(key)

    mid.save(output_file)
    print(f"MIDI file saved to {output_file}")


# ---------------------- CLI ---------------------------------------------- #


def parse_args():
    ap = argparse.ArgumentParser(description="Visual piano transcription")
    ap.add_argument("-i", "--input", required=True, help="Name of input test case")
    ap.add_argument(
        "-c", "--calibrate", action="store_true", help="Force recalculation"
    )
    ap.add_argument("-s", "--show", action="store_true", help="Show debug pictures")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Open input params
    output_file_path = os.path.join("outputs", args.input, "output.json")
    with open(os.path.join("inputs", args.input, "params.json"), "r") as f:
        params = json.load(f)

    # Open video file and extract FPS
    cap = cv2.VideoCapture(os.path.join("inputs", args.input, "video.mp4"))
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}")

    # Whether to recompute the JSON output
    if not os.path.exists(output_file_path) or args.calibrate:
        # Process video and return results
        results = process_video(args, cap, params)
        with open(output_file_path, "w") as f:
            json.dump(results, f, indent=2)
    else:
        with open(output_file_path, "r") as f:
            results = json.load(f)

    # Softening results
    softened = soften_results(results)
    with open(os.path.join("outputs", args.input, "softened.json"), "w") as f:
        json.dump(softened, f, indent=2)

    # Create MIDI file from results
    create_midi_from_results(
        softened, fps, os.path.join("outputs", args.input, "output.mid"), params
    )

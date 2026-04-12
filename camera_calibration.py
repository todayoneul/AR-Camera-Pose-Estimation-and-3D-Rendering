import cv2
import numpy as np
import glob
import os
import json
import sys
from datetime import datetime

# Chessboard configuration (adjust these values based on your chessboard)
CHESSBOARD_SIZE = (11, 7)  # Number of inner corners (columns, rows)
SQUARE_SIZE = 24.0  # Size of each square in mm (adjust based on your printed chessboard)

# Termination criteria for corner refinement
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def prepare_object_points(board_size, square_size):

    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size
    return objp


def extract_frames_from_video(video_path, frame_interval=30):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return [], [], None

    objp = prepare_object_points(CHESSBOARD_SIZE, SQUARE_SIZE)
    obj_points = []  # 3D points in real world space
    img_points = []  # 2D points in image plane
    
    frame_count = 0
    detected_count = 0
    img_size = None
    
    print(f"Processing video: {video_path}")
    print(f"Looking for {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]} chessboard corners...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img_size = gray.shape[::-1]
            
            # Find chessboard corners with additional flags for better detection
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
            ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None, flags=flags)
            
            if ret:
                detected_count += 1
                obj_points.append(objp)
                
                # Refine corner positions for better accuracy
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
                img_points.append(corners_refined)
                
                print(f"  Frame {frame_count}: Chessboard detected! (Total: {detected_count})")
        
        frame_count += 1
    
    cap.release()
    print(f"\nTotal frames processed: {frame_count}")
    print(f"Chessboard detected in: {detected_count} frames")
    
    return obj_points, img_points, img_size


def open_camera_capture(camera_index):

    # On macOS, AVFoundation is the native backend for camera capture.
    if sys.platform == "darwin":
        cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            return cap
        cap.release()

    return cv2.VideoCapture(camera_index)


def extract_frames_from_camera(camera_index=0, frame_interval=15, max_samples=30):

    cap = open_camera_capture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera index: {camera_index}")
        return [], [], None

    objp = prepare_object_points(CHESSBOARD_SIZE, SQUARE_SIZE)
    obj_points = []
    img_points = []

    frame_count = 0
    detected_count = 0
    img_size = None

    print(f"Processing live camera index: {camera_index}")
    print(f"Looking for {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]} chessboard corners...")
    print("Press 'q' to stop capture.")

    read_failures = 0
    max_read_failures = 120

    while True:
        ret, frame = cap.read()
        if not ret:
            read_failures += 1
            if read_failures >= max_read_failures:
                print(
                    "Error: Camera opened but no frames were received. "
                    "Check camera permission and Camo preview state."
                )
                break
            continue

        read_failures = 0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]

        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None, flags=flags)

        view = frame.copy()
        if found:
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
            cv2.drawChessboardCorners(view, CHESSBOARD_SIZE, corners_refined, found)

            if frame_count % frame_interval == 0:
                obj_points.append(objp.copy())
                img_points.append(corners_refined)
                detected_count += 1
                print(f"  Captured sample #{detected_count}")

        cv2.putText(
            view,
            f"Samples: {detected_count}/{max_samples}",
            (10, 28),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            view,
            f"frame interval: {frame_interval} | press q to finish",
            (10, 56),
            cv2.FONT_HERSHEY_DUPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("Camera Calibration Capture", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        frame_count += 1
        if detected_count >= max_samples:
            print(f"Reached max samples: {max_samples}")
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nTotal frames processed: {frame_count}")
    print(f"Chessboard detected in: {detected_count} sampled frames")
    return obj_points, img_points, img_size


def list_available_cameras(max_index=10):

    print(f"Scanning camera indices 0..{max_index - 1}")
    found = []
    for idx in range(max_index):
        cap = open_camera_capture(idx)
        if not cap.isOpened():
            continue

        ret = False
        frame = None
        # Some camera drivers need several reads before a valid frame arrives.
        for _ in range(20):
            ret, frame = cap.read()
            if ret and frame is not None:
                break

        cap.release()
        if not ret or frame is None:
            found.append((idx, None, None))
            print(f"  [{idx}] opened, but no frame yet")
            continue

        h, w = frame.shape[:2]
        found.append((idx, w, h))
        print(f"  [{idx}] {w}x{h}")

    if not found:
        print("No cameras detected.")
    return found


def extract_frames_from_images(image_folder, pattern="*.jpg"):
    objp = prepare_object_points(CHESSBOARD_SIZE, SQUARE_SIZE)
    obj_points = []
    img_points = []
    
    images = glob.glob(os.path.join(image_folder, pattern))
    images.extend(glob.glob(os.path.join(image_folder, "*.png")))
    images.extend(glob.glob(os.path.join(image_folder, "*.jpeg")))
    
    if not images:
        print(f"No images found in {image_folder}")
        return [], [], None
    
    img_size = None
    detected_count = 0
    
    print(f"Processing {len(images)} images from {image_folder}")
    
    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]
        
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        
        if ret:
            detected_count += 1
            obj_points.append(objp)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
            img_points.append(corners_refined)
            print(f"  {os.path.basename(img_path)}: Chessboard detected!")
        else:
            print(f"  {os.path.basename(img_path)}: Chessboard NOT detected")
    
    print(f"\nChessboard detected in: {detected_count}/{len(images)} images")
    
    return obj_points, img_points, img_size


def calibrate_camera(obj_points, img_points, img_size, use_fisheye=False):

    if len(obj_points) < 3:
        print("Error: Need at least 3 valid chessboard images for calibration")
        return None
    
    print("\nPerforming camera calibration...")
    
    if use_fisheye:
        print("Using FISHEYE model for wide-angle lens...")
        # Fisheye calibration
        calibration_flags = (
            cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
            cv2.fisheye.CALIB_CHECK_COND +
            cv2.fisheye.CALIB_FIX_SKEW
        )
        
        # Convert points to the format expected by fisheye calibration
        obj_points_fisheye = [p.reshape(-1, 1, 3).astype(np.float64) for p in obj_points]
        img_points_fisheye = [p.reshape(-1, 1, 2).astype(np.float64) for p in img_points]
        
        camera_matrix = np.zeros((3, 3))
        dist_coeffs = np.zeros((4, 1))
        rvecs = [np.zeros((3, 1)) for _ in range(len(obj_points))]
        tvecs = [np.zeros((3, 1)) for _ in range(len(obj_points))]
        
        try:
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.fisheye.calibrate(
                obj_points_fisheye, img_points_fisheye, img_size,
                camera_matrix, dist_coeffs, rvecs, tvecs,
                calibration_flags,
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
            )
        except cv2.error as e:
            print(f"Fisheye calibration failed: {e}")
            print("Falling back to standard calibration...")
            use_fisheye = False
    
    if not use_fisheye:
        print("Using STANDARD model...")
        # Standard calibration with rational model for better handling of distortion
        calibration_flags = (
            cv2.CALIB_RATIONAL_MODEL +  # Use rational model (k4, k5, k6)
            cv2.CALIB_FIX_PRINCIPAL_POINT  # Fix principal point for stability
        )
        
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, img_size, None, None,
            flags=calibration_flags
        )
    
    # Calculate reprojection error (RMSE)
    total_error = 0
    for i in range(len(obj_points)):
        if use_fisheye:
            img_points_reproj, _ = cv2.fisheye.projectPoints(
                obj_points_fisheye[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
            )
            img_points_reproj = img_points_reproj.reshape(-1, 2)
            error = cv2.norm(img_points_fisheye[i].reshape(-1, 2), img_points_reproj, cv2.NORM_L2) / len(img_points_reproj)
        else:
            img_points_reproj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], 
                                                       camera_matrix, dist_coeffs)
            error = cv2.norm(img_points[i], img_points_reproj, cv2.NORM_L2) / len(img_points_reproj)
        total_error += error ** 2
    
    rmse = np.sqrt(total_error / len(obj_points))
    
    # Extract camera parameters
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    # Distortion coefficients
    dist_flat = dist_coeffs.flatten()
    if use_fisheye:
        # Fisheye: k1, k2, k3, k4
        k1 = float(dist_flat[0]) if len(dist_flat) > 0 else 0.0
        k2 = float(dist_flat[1]) if len(dist_flat) > 1 else 0.0
        k3 = float(dist_flat[2]) if len(dist_flat) > 2 else 0.0
        k4 = float(dist_flat[3]) if len(dist_flat) > 3 else 0.0
        p1, p2 = 0.0, 0.0
    else:
        # Standard: k1, k2, p1, p2, k3 (and possibly more)
        k1 = float(dist_flat[0]) if len(dist_flat) > 0 else 0.0
        k2 = float(dist_flat[1]) if len(dist_flat) > 1 else 0.0
        p1 = float(dist_flat[2]) if len(dist_flat) > 2 else 0.0
        p2 = float(dist_flat[3]) if len(dist_flat) > 3 else 0.0
        k3 = float(dist_flat[4]) if len(dist_flat) > 4 else 0.0
        k4 = 0.0
    
    result = {
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist(),
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "k1": k1,
        "k2": k2,
        "p1": p1,
        "p2": p2,
        "k3": k3,
        "rmse": float(rmse),
        "image_size": list(img_size),
        "num_images_used": len(obj_points),
        "calibration_date": datetime.now().isoformat(),
        "model": "fisheye" if use_fisheye else "standard",
        "rvecs": [r.flatten().tolist() for r in rvecs],
        "tvecs": [t.flatten().tolist() for t in tvecs]
    }
    
    return result


def save_calibration_result(result, output_path="calibration_result.json"):
    """
    Save calibration result to JSON file.
    """
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=4)
    print(f"\nCalibration result saved to: {output_path}")


def print_calibration_result(result):
    """
    Print calibration results in a formatted way.
    """
    print("\n" + "="*60)
    print("CAMERA CALIBRATION RESULTS")
    print("="*60)
    model = result.get('model', 'standard')
    print(f"\nModel: {model.upper()}")
    print(f"\nIntrinsic Parameters:")
    print(f"  fx (focal length x): {result['fx']:.4f} pixels")
    print(f"  fy (focal length y): {result['fy']:.4f} pixels")
    print(f"  cx (principal point x): {result['cx']:.4f} pixels")
    print(f"  cy (principal point y): {result['cy']:.4f} pixels")
    print(f"\nDistortion Coefficients:")
    print(f"  k1: {result['k1']:.6f}")
    print(f"  k2: {result['k2']:.6f}")
    if model != 'fisheye':
        print(f"  p1: {result['p1']:.6f}")
        print(f"  p2: {result['p2']:.6f}")
    print(f"  k3: {result['k3']:.6f}")
    print(f"\nCalibration Quality:")
    print(f"  RMSE (Reprojection Error): {result['rmse']:.6f} pixels")
    print(f"  Number of images used: {result['num_images_used']}")
    print(f"  Image size: {result['image_size'][0]}x{result['image_size'][1]}")
    print("="*60)


def generate_readme_content(result):
    """
    Generate README.md content with calibration results.
    """
    content = f"""## Camera Calibration Results

### Intrinsic Parameters (Camera Matrix)
| Parameter | Value |
|-----------|-------|
| fx (focal length x) | {result['fx']:.4f} pixels |
| fy (focal length y) | {result['fy']:.4f} pixels |
| cx (principal point x) | {result['cx']:.4f} pixels |
| cy (principal point y) | {result['cy']:.4f} pixels |

### Distortion Coefficients
| Parameter | Value |
|-----------|-------|
| k1 (radial) | {result['k1']:.6f} |
| k2 (radial) | {result['k2']:.6f} |
| p1 (tangential) | {result['p1']:.6f} |
| p2 (tangential) | {result['p2']:.6f} |
| k3 (radial) | {result['k3']:.6f} |

### Calibration Quality
- **RMSE (Reprojection Error)**: {result['rmse']:.6f} pixels
- **Number of images used**: {result['num_images_used']}
- **Image size**: {result['image_size'][0]}x{result['image_size'][1]}
"""
    return content


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Camera Calibration using Chessboard Pattern')
    parser.add_argument('input', nargs='?', help='Path to video file, image folder, or camera index (e.g., 0)')
    parser.add_argument('--board-size', type=str, default='11,7',
                        help='Chessboard inner corners (columns,rows), default: 11,7')
    parser.add_argument('--square-size', type=float, default=24.0,
                        help='Size of chessboard square in mm, default: 24.0')
    parser.add_argument('--frame-interval', type=int, default=15,
                        help='Frame interval for video processing, default: 15')
    parser.add_argument('--max-samples', type=int, default=30,
                        help='Max sampled chessboard detections in camera mode, default: 30')
    parser.add_argument('--output', type=str, default='calibration_result.json',
                        help='Output JSON file path')
    parser.add_argument('--fisheye', action='store_true',
                        help='Use fisheye model for wide-angle/action cameras')
    parser.add_argument('--list-cameras', action='store_true',
                        help='List available camera indices and exit (or continue if input is given)')
    
    args = parser.parse_args()

    if args.list_cameras:
        list_available_cameras()
        if args.input is None:
            return

    if args.input is None:
        parser.error('input is required unless --list-cameras is used alone')
    
    # Update global settings
    global CHESSBOARD_SIZE, SQUARE_SIZE
    board_size = args.board_size.split(',')
    CHESSBOARD_SIZE = (int(board_size[0]), int(board_size[1]))
    SQUARE_SIZE = args.square_size
    
    # Determine input type (camera, video, or image folder)
    if args.input.isdigit():
        obj_points, img_points, img_size = extract_frames_from_camera(
            int(args.input), args.frame_interval, args.max_samples
        )
    elif os.path.isfile(args.input):
        # Video file
        obj_points, img_points, img_size = extract_frames_from_video(
            args.input, args.frame_interval
        )
    elif os.path.isdir(args.input):
        # Image folder
        obj_points, img_points, img_size = extract_frames_from_images(args.input)
    else:
        print(f"Error: Input path does not exist: {args.input}")
        return
    
    if len(obj_points) == 0:
        print("Error: No valid chessboard images found!")
        return
    
    # Perform calibration
    result = calibrate_camera(obj_points, img_points, img_size, use_fisheye=args.fisheye)
    
    if result:
        print_calibration_result(result)
        save_calibration_result(result, args.output)
        
        # Print README content for easy copy-paste
        print("\n" + "="*60)
        print("README.md Content (copy below):")
        print("="*60)
        print(generate_readme_content(result))


if __name__ == "__main__":
    main()

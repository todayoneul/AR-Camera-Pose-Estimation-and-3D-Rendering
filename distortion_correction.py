import cv2
import numpy as np
import json
import os
import argparse


def create_compatible_video_writer(path, fps, frame_size):
    codecs = ['avc1', 'H264', 'mp4v']
    for codec in codecs:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(path, fourcc, fps, frame_size)
        if writer.isOpened():
            print(f"Using codec '{codec}' for: {os.path.basename(path)}")
            return writer
        writer.release()

    raise RuntimeError(f"Could not open VideoWriter for: {path}")


def load_calibration(calibration_path):
    with open(calibration_path, 'r') as f:
        result = json.load(f)
    
    camera_matrix = np.array(result['camera_matrix'])
    dist_coeffs = np.array(result['dist_coeffs'])
    model = result.get('model', 'standard')
    
    return camera_matrix, dist_coeffs, result, model


def undistort_image(image, camera_matrix, dist_coeffs, alpha=1.0, model='standard'):
    h, w = image.shape[:2]
    
    if model == 'fisheye':
        # Fisheye undistortion
        new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            camera_matrix, dist_coeffs, (w, h), np.eye(3), balance=alpha
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, np.eye(3), new_camera_matrix, (w, h), cv2.CV_16SC2
        )
        undistorted = cv2.remap(image, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    else:
        # Standard undistortion
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), alpha, (w, h)
        )
        undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
        
        # Optionally crop to ROI
        if alpha < 1.0:
            x, y, w_roi, h_roi = roi
            if w_roi > 0 and h_roi > 0:
                undistorted = undistorted[y:y+h_roi, x:x+w_roi]
    
    return undistorted, new_camera_matrix


def undistort_image_with_map(image, camera_matrix, dist_coeffs, alpha=1.0, map1=None, map2=None, model='standard'):
    h, w = image.shape[:2]
    
    if map1 is None or map2 is None:
        if model == 'fisheye':
            new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                camera_matrix, dist_coeffs, (w, h), np.eye(3), balance=alpha
            )
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                camera_matrix, dist_coeffs, np.eye(3), new_camera_matrix, (w, h), cv2.CV_16SC2
            )
        else:
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                camera_matrix, dist_coeffs, (w, h), alpha, (w, h)
            )
            map1, map2 = cv2.initUndistortRectifyMap(
                camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_16SC2
            )
    
    undistorted = cv2.remap(image, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    return undistorted, map1, map2


def create_comparison_image(original, undistorted, direction='horizontal'):
    # Resize if dimensions don't match
    if original.shape != undistorted.shape:
        undistorted = cv2.resize(undistorted, (original.shape[1], original.shape[0]))
    
    # Add labels
    h, w = original.shape[:2]
    original_labeled = original.copy()
    undistorted_labeled = undistorted.copy()
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(original_labeled, "Original (Distorted)", (10, 30), 
                font, 0.8, (0, 255, 0), 2)
    cv2.putText(undistorted_labeled, "Corrected (Undistorted)", (10, 30), 
                font, 0.8, (0, 255, 0), 2)
    
    if direction == 'horizontal':
        comparison = np.hstack([original_labeled, undistorted_labeled])
    else:
        comparison = np.vstack([original_labeled, undistorted_labeled])
    
    return comparison


def process_image(input_path, output_folder, camera_matrix, dist_coeffs, alpha=0.6, model='standard', crop_roi=True):
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image: {input_path}")
        return
    
    # Undistort
    h, w = img.shape[:2]
    
    if model == 'fisheye':
        new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            camera_matrix, dist_coeffs, (w, h), np.eye(3), balance=alpha
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, np.eye(3), new_camera_matrix, (w, h), cv2.CV_16SC2
        )
        undistorted = cv2.remap(img, map1, map2, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
        roi = None
    else:
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), alpha, (w, h)
        )
        map1, map2 = cv2.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_16SC2
        )
        undistorted = cv2.remap(img, map1, map2, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
    
    # Crop ROI if enabled
    if crop_roi and roi is not None and roi[2] > 0 and roi[3] > 0:
        x, y, roi_w, roi_h = roi
        undistorted = undistorted[y:y+roi_h, x:x+roi_w]
    
    # Create comparison
    comparison = create_comparison_image(img, undistorted)
    
    # Save results
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    undistorted_path = os.path.join(output_folder, f"{base_name}_undistorted.jpg")
    comparison_path = os.path.join(output_folder, f"{base_name}_comparison.jpg")
    
    cv2.imwrite(undistorted_path, undistorted)
    cv2.imwrite(comparison_path, comparison)
    
    print(f"Saved: {undistorted_path}")
    print(f"Saved: {comparison_path}")
    
    return comparison_path


def process_video(input_path, output_folder, camera_matrix, dist_coeffs, alpha=0.6, save_comparison=True, model='standard', crop_roi=True):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {input_path}")
        return
    
    # Get video properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video: {input_path}")
    print(f"Resolution: {w}x{h}, FPS: {fps}, Total frames: {total_frames}")
    print(f"Model: {model}, Alpha: {alpha}")
    
    # Prepare output
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    # Precompute undistortion maps and ROI for faster processing
    if model == 'fisheye':
        new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            camera_matrix, dist_coeffs, (w, h), np.eye(3), balance=alpha
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, np.eye(3), new_camera_matrix, (w, h), cv2.CV_16SC2
        )
        roi = None
    else:
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), alpha, (w, h)
        )
        map1, map2 = cv2.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_16SC2
        )
    
    # Determine output size based on ROI
    output_w, output_h = w, h
    if crop_roi and roi is not None and roi[2] > 0 and roi[3] > 0:
        x, y, roi_w, roi_h = roi
        output_w, output_h = roi_w, roi_h
        print(f"Cropping to ROI: {roi_w}x{roi_h} (removing black borders)")
    
    # Undistorted video
    undistorted_path = os.path.join(output_folder, f"{base_name}_undistorted.mp4")
    out_undistorted = create_compatible_video_writer(undistorted_path, fps, (output_w, output_h))
    
    # Comparison video
    if save_comparison:
        comparison_path = os.path.join(output_folder, f"{base_name}_comparison.mp4")
        out_comparison = create_compatible_video_writer(comparison_path, fps, (output_w * 2, output_h))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Undistort using precomputed maps with better interpolation
        undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
        
        # Crop ROI if enabled
        frame_for_comparison = frame
        if crop_roi and roi is not None and roi[2] > 0 and roi[3] > 0:
            x, y, roi_w, roi_h = roi
            undistorted = undistorted[y:y+roi_h, x:x+roi_w]
            frame_for_comparison = frame[y:y+roi_h, x:x+roi_w]  # Crop original frame too
        
        out_undistorted.write(undistorted)
        
        if save_comparison:
            comparison = create_comparison_image(frame_for_comparison, undistorted)
            out_comparison.write(comparison)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"  Processed {frame_count}/{total_frames} frames...")
    
    cap.release()
    out_undistorted.release()
    if save_comparison:
        out_comparison.release()
    
    print(f"\nSaved undistorted video: {undistorted_path}")
    if save_comparison:
        print(f"Saved comparison video: {comparison_path}")
        
        # Also save a comparison frame as image for README
        cap = cv2.VideoCapture(input_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)  # Middle frame
        ret, frame = cap.read()
        if ret:
            undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
            # Crop both frames if needed
            frame_for_comparison = frame
            if crop_roi and roi is not None and roi[2] > 0 and roi[3] > 0:
                x, y, roi_w, roi_h = roi
                undistorted = undistorted[y:y+roi_h, x:x+roi_w]
                frame_for_comparison = frame[y:y+roi_h, x:x+roi_w]
            comparison = create_comparison_image(frame_for_comparison, undistorted)
            comparison_img_path = os.path.join(output_folder, f"{base_name}_comparison_frame.jpg")
            cv2.imwrite(comparison_img_path, comparison)
            print(f"Saved comparison frame: {comparison_img_path}")
        cap.release()
        
        return comparison_path
    
    return undistorted_path


def live_preview(camera_matrix, dist_coeffs, camera_id=0, alpha=0.6, model='standard'):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return
    
    # Get camera resolution
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Precompute undistortion maps
    if model == 'fisheye':
        new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            camera_matrix, dist_coeffs, (w, h), np.eye(3), balance=alpha
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, np.eye(3), new_camera_matrix, (w, h), cv2.CV_16SC2
        )
        roi = None
    else:
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), alpha, (w, h)
        )
        map1, map2 = cv2.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_16SC2
        )
    
    print("Live preview started. Press 'q' to quit, 's' to save a screenshot.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
        
        # Crop ROI to remove black borders
        if roi is not None and roi[2] > 0 and roi[3] > 0:
            x, y, roi_w, roi_h = roi
            undistorted = undistorted[y:y+roi_h, x:x+roi_w]
        
        comparison = create_comparison_image(frame, undistorted)
        
        # Resize for display if too large
        display = comparison
        if comparison.shape[1] > 1920:
            scale = 1920 / comparison.shape[1]
            display = cv2.resize(comparison, None, fx=scale, fy=scale)
        
        cv2.imshow('Distortion Correction (Press Q to quit, S to save)', display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            screenshot_path = 'distortion_correction_screenshot.jpg'
            cv2.imwrite(screenshot_path, comparison)
            print(f"Saved screenshot: {screenshot_path}")
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Lens Distortion Correction')
    parser.add_argument('calibration', help='Path to calibration_result.json')
    parser.add_argument('--input', '-i', help='Path to image or video file (optional for live preview)')
    parser.add_argument('--output', '-o', default='output', help='Output folder')
    parser.add_argument('--alpha', type=float, default=0.6,
                        help='Scaling parameter (0-1). Default: 0.6 (recommended)')
    parser.add_argument('--strength', type=float, default=1.0,
                        help='Distortion correction strength. <1.0 reduces correction (e.g. 0.9), 1.0=full')
    parser.add_argument('--crop', action='store_true', default=True,
                        help='Crop black borders (default: True)')
    parser.add_argument('--no-crop', dest='crop', action='store_false',
                        help='Do not crop black borders')
    parser.add_argument('--live', action='store_true', help='Live preview from webcam')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID for live preview')
    
    args = parser.parse_args()
    
    # Load calibration
    if not os.path.exists(args.calibration):
        print(f"Error: Calibration file not found: {args.calibration}")
        return
    
    camera_matrix, dist_coeffs, calib_result, model = load_calibration(args.calibration)

    # Optional scaling to soften over-correction while keeping calibration fixed.
    if model == 'standard' and args.strength != 1.0:
        dist_coeffs = dist_coeffs.astype(np.float64) * args.strength

    print(f"Loaded calibration from: {args.calibration}")
    print(f"  Model: {model}")
    print(f"  fx={calib_result['fx']:.2f}, fy={calib_result['fy']:.2f}")
    print(f"  cx={calib_result['cx']:.2f}, cy={calib_result['cy']:.2f}")
    print(f"  RMSE={calib_result['rmse']:.6f}")
    if model == 'standard':
        print(f"  Strength={args.strength:.3f}")
    
    # Create output folder
    os.makedirs(args.output, exist_ok=True)
    
    if args.live:
        # Live preview mode
        live_preview(camera_matrix, dist_coeffs, args.camera, args.alpha, model)
    elif args.input:
        # Process file
        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}")
            return
        
        # Determine file type
        ext = os.path.splitext(args.input)[1].lower()
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        if ext in video_extensions:
            process_video(args.input, args.output, camera_matrix, dist_coeffs, args.alpha, model=model, crop_roi=args.crop)
        elif ext in image_extensions:
            process_image(args.input, args.output, camera_matrix, dist_coeffs, args.alpha, model=model, crop_roi=args.crop)
        else:
            print(f"Error: Unsupported file format: {ext}")
    else:
        print("Error: Please provide --input or --live option")
        parser.print_help()


if __name__ == "__main__":
    main()

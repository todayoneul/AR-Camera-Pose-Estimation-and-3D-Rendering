import argparse
import json
import os

import cv2
import numpy as np


def parse_pair_int(text):
    values = [int(v.strip()) for v in text.split(",")]
    if len(values) != 2:
        raise ValueError("Expected two comma-separated integers, e.g., 11,7")
    return values[0], values[1]


def parse_pair_float(text):
    values = [float(v.strip()) for v in text.split(",")]
    if len(values) != 2:
        raise ValueError("Expected two comma-separated floats, e.g., 4,2")
    return values[0], values[1]


def load_calibration(calibration_path, frame_width, frame_height):
    if calibration_path and os.path.exists(calibration_path):
        with open(calibration_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        camera_matrix = np.array(result["camera_matrix"], dtype=np.float64)
        dist_coeffs = np.array(result["dist_coeffs"], dtype=np.float64)
        print(f"Loaded calibration: {calibration_path}")
        return camera_matrix, dist_coeffs

    fx = frame_width * 0.9
    fy = frame_width * 0.9
    cx = frame_width / 2.0
    cy = frame_height / 2.0
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)
    print("Calibration file not found. Using approximated camera matrix.")
    return camera_matrix, dist_coeffs


def open_capture(input_source):
    if input_source.isdigit():
        return cv2.VideoCapture(int(input_source))
    return cv2.VideoCapture(input_source)


def create_writer_if_needed(path, fps, width, height):
    if not path:
        return None

    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fps = float(fps) if fps > 0 else 30.0
    size = (int(width), int(height))

    root, ext = os.path.splitext(path)
    ext = ext.lower()
    candidates = [
        (path, "mp4v"),
        (path, "avc1"),
        (path if ext == ".avi" else f"{root}.avi", "MJPG"),
    ]

    for out_path, codec in candidates:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(out_path, fourcc, fps, size)
        if writer.isOpened():
            print(f"Saving AR demo: {out_path} (codec={codec})")
            return writer
        writer.release()

    print("Warning: Could not open output writer. Skipping video save.")
    return None


def create_board_object_points(board_size, square_size_m):
    cols, rows = board_size
    obj = np.array([[c, r, 0.0] for r in range(rows) for c in range(cols)], dtype=np.float32)
    obj *= square_size_m
    return obj


def create_box(origin_cell, size_cell, square_size_m):
    ox, oy = origin_cell
    sx, sy, sz = size_cell

    lower = square_size_m * np.array(
        [
            [ox, oy, 0.0],
            [ox + sx, oy, 0.0],
            [ox + sx, oy + sy, 0.0],
            [ox, oy + sy, 0.0],
        ],
        dtype=np.float32,
    )
    upper = square_size_m * np.array(
        [
            [ox, oy, -sz],
            [ox + sx, oy, -sz],
            [ox + sx, oy + sy, -sz],
            [ox, oy + sy, -sz],
        ],
        dtype=np.float32,
    )
    return lower, upper


def draw_box(frame, lower_pts, upper_pts):
    lower_i = np.int32(lower_pts.reshape(-1, 2))
    upper_i = np.int32(upper_pts.reshape(-1, 2))

    cv2.polylines(frame, [lower_i], True, (255, 0, 0), 2)
    cv2.polylines(frame, [upper_i], True, (0, 0, 255), 2)
    for b, t in zip(lower_i, upper_i):
        cv2.line(frame, tuple(b), tuple(t), (0, 255, 0), 2)


def load_obj(filename):
    vertices = []
    faces = []
    try:
        with open(filename, "r") as f:
            for line in f:
                if line.startswith("v "):
                    v = line.split()[1:4]
                    vertices.append([float(v[0]), float(v[1]), float(v[2])])
                elif line.startswith("f "):
                    f_idx = []
                    for vertex_data in line.split()[1:]:
                        idx = int(vertex_data.split("/")[0])
                        f_idx.append(idx - 1)
                    faces.append(f_idx)
        return np.array(vertices, dtype=np.float32), faces
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None, None


def camera_position_text(rvec, tvec):
    rmat, _ = cv2.Rodrigues(rvec)
    pos = (-rmat.T @ tvec).flatten()
    return f"XYZ: [{pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}]"


def detect_chessboard(gray, board_size, flags, criteria, use_sb):
    if use_sb and hasattr(cv2, "findChessboardCornersSB"):
        sb_flags = cv2.CALIB_CB_NORMALIZE_IMAGE
        if hasattr(cv2, "CALIB_CB_EXHAUSTIVE"):
            sb_flags |= cv2.CALIB_CB_EXHAUSTIVE
        found_sb, corners_sb = cv2.findChessboardCornersSB(gray, board_size, flags=sb_flags)
        if found_sb:
            return True, corners_sb.astype(np.float32)

    found, corners = cv2.findChessboardCorners(gray, board_size, flags=flags)
    if not found:
        return False, None

    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners


def solve_pose_from_points(obj_points, img_points, camera_matrix, dist_coeffs, reproj_error):
    obj_points = np.asarray(obj_points, dtype=np.float32).reshape(-1, 3)
    img_points = np.asarray(img_points, dtype=np.float32).reshape(-1, 2)

    if len(obj_points) < 6 or len(img_points) < 6:
        return False, None, None, 0

    # RANSAC 제거 (평면 모델에서 무작위 4점 추출 시 발생하는 모호성(Planar Ambiguity) 방지)
    # 체스보드 코너처럼 전역적으로 일관된 모델에서는 전체 점을 활용한 일반 solvePnP가 훨씬 더 강력함.
    ok, rvec, tvec = cv2.solvePnP(
        obj_points,
        img_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not ok:
        return False, None, None, 0

    if not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(tvec)):
        return False, None, None, 0

    if float(np.asarray(tvec).reshape(-1)[2]) <= 0:
        return False, None, None, 0

    return True, rvec, tvec, len(img_points)


def smooth_pose(prev_rvec, prev_tvec, rvec, tvec, alpha):
    if prev_rvec is None or prev_tvec is None:
        return rvec, tvec

    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha <= 0.0:
        return rvec, tvec

    if not np.all(np.isfinite(prev_rvec)) or not np.all(np.isfinite(prev_tvec)):
        return rvec, tvec
    if not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(tvec)):
        return prev_rvec, prev_tvec

    prev_rmat, _ = cv2.Rodrigues(prev_rvec)
    curr_rmat, _ = cv2.Rodrigues(rvec)
    blended_rmat = (1.0 - alpha) * prev_rmat + alpha * curr_rmat
    if not np.all(np.isfinite(blended_rmat)):
        return rvec, tvec

    try:
        u, _, vt = np.linalg.svd(blended_rmat)
    except np.linalg.LinAlgError:
        return rvec, tvec

    blended_rmat = u @ vt
    if np.linalg.det(blended_rmat) < 0:
        blended_rmat = u @ np.diag([1.0, 1.0, -1.0]) @ vt

    blended_rvec, _ = cv2.Rodrigues(blended_rmat)
    blended_tvec = (1.0 - alpha) * prev_tvec + alpha * tvec
    return blended_rvec, blended_tvec


def pose_jump_metrics(prev_rvec, prev_tvec, rvec, tvec):
    prev_rmat, _ = cv2.Rodrigues(prev_rvec)
    curr_rmat, _ = cv2.Rodrigues(rvec)
    r_delta = curr_rmat @ prev_rmat.T
    cos_theta = np.clip((np.trace(r_delta) - 1.0) * 0.5, -1.0, 1.0)
    angle_deg = float(np.degrees(np.arccos(cos_theta)))
    trans_dist = float(
        np.linalg.norm(np.asarray(tvec).reshape(-1) - np.asarray(prev_tvec).reshape(-1))
    )
    return angle_deg, trans_dist


def is_box_projection_valid(
    box_lower,
    box_upper,
    lower_2d,
    upper_2d,
    rvec,
    tvec,
    frame_shape,
    min_box_depth,
    max_box_edge_ratio,
    max_box_frame_mult,
):
    pts = np.vstack([lower_2d.reshape(-1, 2), upper_2d.reshape(-1, 2)])
    if not np.all(np.isfinite(pts)):
        return False

    h, w = frame_shape[:2]
    lim = max(h, w) * float(max_box_frame_mult)
    if np.max(np.abs(pts)) > lim:
        return False

    rmat, _ = cv2.Rodrigues(rvec)
    world_pts = np.vstack([box_lower, box_upper]).T
    cam_pts = (rmat @ world_pts) + np.asarray(tvec).reshape(3, 1)
    min_depth = float(np.min(cam_pts[2]))
    if min_depth <= float(min_box_depth):
        return False

    lower = lower_2d.reshape(-1, 2)
    upper = upper_2d.reshape(-1, 2)
    edges = []
    for quad in (lower, upper):
        for i in range(4):
            edges.append(np.linalg.norm(quad[i] - quad[(i + 1) % 4]))
    for i in range(4):
        edges.append(np.linalg.norm(lower[i] - upper[i]))

    edges = np.asarray(edges, dtype=np.float64)
    if np.any(~np.isfinite(edges)):
        return False
    min_edge = float(np.min(edges))
    max_edge = float(np.max(edges))
    if min_edge < 1e-3:
        return False
    if max_edge / min_edge > float(max_box_edge_ratio):
        return False

    return True


def draw_pose_overlay(
    frame,
    box_lower,
    box_upper,
    obj_vertices,
    obj_faces,
    solid_render,
    rvec,
    tvec,
    camera_matrix,
    dist_coeffs,
):
    if obj_vertices is not None and obj_faces is not None:
        imgpts, _ = cv2.projectPoints(obj_vertices, rvec, tvec, camera_matrix, dist_coeffs)
        imgpts = np.int32(imgpts).reshape(-1, 2)
        
        if solid_render:
            # Painter's Algorithm with basic shading
            R, _ = cv2.Rodrigues(rvec)
            # Transform vertices to camera space for depth sorting and normal calculation
            pts_cam = (R @ obj_vertices.T).T + tvec.T
            
            # Extract only valid triangular/quad faces
            valid_faces = [f for f in obj_faces if len(f) >= 3]
            
            # Simple Z sorting (average Z of each face)
            face_depths = np.array([np.mean(pts_cam[f, 2]) for f in valid_faces])
            sort_idx = np.argsort(face_depths)[::-1] # farthest to nearest
            
            # Pre-calculate normals for simple lighting (diffuse)
            light_dir = np.array([0.0, 0.0, -1.0]) # light coming from camera
            
            base_color = np.array([160, 160, 160]) # BGR color for the model (Gray)
            
            for idx in sort_idx:
                f = valid_faces[idx]
                # Normal calculation for lighting
                v0, v1, v2 = pts_cam[f[0]], pts_cam[f[1]], pts_cam[f[2]]
                normal = np.cross(v1 - v0, v2 - v0)
                norm_len = np.linalg.norm(normal)
                if norm_len > 0:
                    normal /= norm_len
                    # 면이 뒤집혀 있는(법선 벡터 반대) 경우를 감안해 절댓값 적용
                    intensity = abs(np.dot(normal, light_dir))
                else:
                    intensity = 0.5
                
                # Ambient + Diffuse
                final_color = base_color * (0.3 + 0.7 * intensity)
                final_color = tuple(map(int, final_color))
                
                poly = imgpts[f]
                cv2.fillPoly(frame, [poly], final_color)
                # 경계선을 살짝 그려주면 렌더링이 훨씬 깔끔해짐
                cv2.polylines(frame, [poly], True, (50, 50, 50), 1)
                
        else:
            cv2.polylines(frame, [imgpts[f] for f in obj_faces], True, (0, 255, 255), 1)
    else:
        lower_2d, _ = cv2.projectPoints(box_lower, rvec, tvec, camera_matrix, dist_coeffs)
        upper_2d, _ = cv2.projectPoints(box_upper, rvec, tvec, camera_matrix, dist_coeffs)
        draw_box(frame, lower_2d, upper_2d)

    cv2.putText(
        frame,
        camera_position_text(rvec, tvec),
        (10, 55),
        cv2.FONT_HERSHEY_DUPLEX,
        0.6,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    return True


def main():
    parser = argparse.ArgumentParser(description="Pose Estimation + AR from chessboard")
    parser.add_argument("--input", default="0", help="Input video path or camera id")
    parser.add_argument("--calibration", default="calibration_result.json", help="Calibration JSON path")
    parser.add_argument("--board-size", default="11,7", help="Chessboard inner corners: cols,rows")
    parser.add_argument("--square-size", type=float, default=0.024, help="Square size in meters")
    parser.add_argument("--box-origin", default="4,2", help="Box origin in board cells: x,y")
    parser.add_argument("--box-size", default="1,2,1", help="Box size in board cells: sx,sy,sz")
    parser.add_argument("--obj", default="", help="Path to .obj file to render instead of AR box")
    parser.add_argument("--obj-scale", type=float, default=1.0, help="OBJ model scaling multiplier")
    parser.add_argument("--obj-rx", type=float, default=0.0, help="Rotate OBJ around X-axis (degrees)")
    parser.add_argument("--obj-ry", type=float, default=0.0, help="Rotate OBJ around Y-axis (degrees)")
    parser.add_argument("--obj-rz", type=float, default=0.0, help="Rotate OBJ around Z-axis (degrees)")
    parser.add_argument("--solid", action="store_true", help="Render solid shaded faces instead of wireframe (Slow!)")
    parser.add_argument("--flip-x", action="store_true", help="Flip X-axis")
    parser.add_argument("--flip-y", action="store_true", help="Flip Y-axis")
    parser.add_argument("--flip-z", action="store_true", help="Flip Z-axis (useful to force objects to extrude upwards)")
    parser.add_argument("--output", default="", help="Optional output video path")
    parser.add_argument("--compare-view", action="store_true", help="Show/save side-by-side original|AR")
    parser.add_argument("--max-frames", type=int, default=0, help="Process only first N frames (0 = all)")
    parser.add_argument(
        "--pnp-reproj",
        type=float,
        default=5.0,
        help="PnP RANSAC reprojection threshold in pixels",
    )
    parser.add_argument(
        "--min-track-corners",
        type=int,
        default=20,
        help="Minimum tracked corners to keep optical-flow fallback",
    )
    parser.add_argument(
        "--pose-smooth",
        type=float,
        default=0.2,
        help="Pose smoothing factor in [0,1]",
    )
    parser.add_argument(
        "--hold-frames",
        type=int,
        default=8,
        help="Keep last pose this many frames after temporary loss",
    )
    parser.add_argument(
        "--no-flow-fallback",
        action="store_true",
        help="Disable optical-flow fallback when chessboard detection fails",
    )
    parser.add_argument(
        "--use-sb",
        action="store_true",
        help="Use findChessboardCornersSB first (slower but often more robust)",
    )
    parser.add_argument(
        "--flow-fb-thresh",
        type=float,
        default=1.5,
        help="Forward-backward optical-flow error threshold in pixels",
    )
    parser.add_argument(
        "--max-angle-jump",
        type=float,
        default=25.0,
        help="Reject pose update if rotation jump exceeds this value (degrees)",
    )
    parser.add_argument(
        "--max-trans-jump",
        type=float,
        default=0.25,
        help="Reject pose update if translation jump exceeds this value (meters)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable GUI display windows (useful for batch parameter tests)",
    )
    parser.add_argument(
        "--draw-corners",
        action="store_true",
        help="Draw full chessboard corner lines (can look cluttered)",
    )
    parser.add_argument(
        "--min-box-depth",
        type=float,
        default=0.01,
        help="Reject box rendering when any box corner depth is below this value (meters)",
    )
    parser.add_argument(
        "--max-box-edge-ratio",
        type=float,
        default=30.0,
        help="Reject rendering when box edge length ratio exceeds this value",
    )
    parser.add_argument(
        "--max-box-frame-mult",
        type=float,
        default=2.0,
        help="Reject rendering when projected points are farther than this * frame size",
    )
    args = parser.parse_args()

    board_size = parse_pair_int(args.board_size)
    box_origin = parse_pair_float(args.box_origin)
    box_size_vals = [float(v.strip()) for v in args.box_size.split(",")]
    if len(box_size_vals) != 3:
        raise ValueError("--box-size must be sx,sy,sz")

    cap = open_capture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input source: {args.input}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-3:
        fps = 30.0

    camera_matrix, dist_coeffs = load_calibration(args.calibration, frame_w, frame_h)
    obj_points = create_board_object_points(board_size, args.square_size)
    box_lower, box_upper = create_box(box_origin, box_size_vals, args.square_size)

    obj_vertices, obj_faces = None, None
    if args.obj and os.path.exists(args.obj):
        obj_vertices, obj_faces = load_obj(args.obj)
        if obj_vertices is not None:
            obj_vertices = obj_vertices * args.obj_scale
            
            if args.flip_x:
                obj_vertices[:, 0] *= -1
            if args.flip_y:
                obj_vertices[:, 1] *= -1
            if args.flip_z:
                obj_vertices[:, 2] *= -1

            # 회전 적용 (Rx, Ry, Rz)
            rx, ry, rz = np.radians(args.obj_rx), np.radians(args.obj_ry), np.radians(args.obj_rz)
            
            # X 회전
            cos_x, sin_x = np.cos(rx), np.sin(rx)
            R_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
            # Y 회전
            cos_y, sin_y = np.cos(ry), np.sin(ry)
            R_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
            # Z 회전
            cos_z, sin_z = np.cos(rz), np.sin(rz)
            R_z = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])
            
            R_total = R_z @ R_y @ R_x
            obj_vertices = (R_total @ obj_vertices.T).T

            ox, oy = box_origin
            obj_vertices[:, 0] += ox * args.square_size
            obj_vertices[:, 1] += oy * args.square_size

    output_w = frame_w * 2 if args.compare_view else frame_w
    writer = create_writer_if_needed(args.output, fps, output_w, frame_h)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

    print("Press 'q' to quit.")

    processed = 0
    tracked = 0
    tracked_by_corners = 0
    tracked_by_flow = 0
    tracked_by_hold = 0
    jump_reject_count = 0
    render_reject_count = 0

    prev_gray = None
    tracked_points = None
    tracked_ids = None
    last_rvec = None
    last_tvec = None
    lost_count = 0

    flow_criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT,
        30,
        0.01,
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        processed += 1
        if args.max_frames > 0 and processed > args.max_frames:
            break

        frame_aug = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = detect_chessboard(gray, board_size, flags, criteria, args.use_sb)

        status = "Tracking lost"
        color = (0, 0, 255)
        tracked_this_frame = False

        rvec = None
        tvec = None

        if found:
            ok_pnp, rvec, tvec, pnp_inliers = solve_pose_from_points(
                obj_points,
                corners,
                camera_matrix,
                dist_coeffs,
                args.pnp_reproj,
            )

            if ok_pnp:
                if last_rvec is not None:
                    angle_deg, trans_dist = pose_jump_metrics(last_rvec, last_tvec, rvec, tvec)
                    if (
                        args.max_angle_jump > 0
                        and angle_deg > args.max_angle_jump
                    ) or (
                        args.max_trans_jump > 0
                        and trans_dist > args.max_trans_jump
                    ):
                        ok_pnp = False
                        jump_reject_count += 1

            if ok_pnp:
                if last_rvec is not None and args.pose_smooth > 0:
                    rvec, tvec = smooth_pose(last_rvec, last_tvec, rvec, tvec, args.pose_smooth)

                last_rvec = rvec.copy()
                last_tvec = tvec.copy()
                lost_count = 0
                tracked_this_frame = True
                tracked += 1
                tracked_by_corners += 1

                draw_pose_overlay(frame_aug, box_lower, box_upper, obj_vertices, obj_faces, args.solid, rvec, tvec, camera_matrix, dist_coeffs)
                if True:
                    if args.draw_corners:
                        cv2.drawChessboardCorners(frame_aug, board_size, corners, found)
                    else:
                        for p in corners.reshape(-1, 2):
                            cv2.circle(frame_aug, tuple(np.int32(p)), 2, (0, 255, 255), -1)

                    tracked_points = corners.reshape(-1, 1, 2).astype(np.float32)
                    tracked_ids = np.arange(len(tracked_points), dtype=np.int32)

                    status = f"Tracked(corners) | corners={len(corners)} | inliers={pnp_inliers}"
                    color = (0, 255, 0)

        elif (
            not args.no_flow_fallback
            and prev_gray is not None
            and tracked_points is not None
            and tracked_ids is not None
            and len(tracked_points) >= args.min_track_corners
        ):
            next_points, st, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                gray,
                tracked_points,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=flow_criteria,
            )

            if next_points is not None and st is not None:
                back_points, st_back, _ = cv2.calcOpticalFlowPyrLK(
                    gray,
                    prev_gray,
                    next_points,
                    None,
                    winSize=(21, 21),
                    maxLevel=3,
                    criteria=flow_criteria,
                )

                st = st.reshape(-1).astype(bool)
                if back_points is not None and st_back is not None:
                    st_back = st_back.reshape(-1).astype(bool)
                    fb_err = np.linalg.norm(
                        tracked_points.reshape(-1, 2) - back_points.reshape(-1, 2),
                        axis=1,
                    )
                    st = st & st_back & (fb_err <= float(args.flow_fb_thresh))

                flow_points = next_points.reshape(-1, 2)[st]
                flow_ids = tracked_ids[st]

                if len(flow_points) >= args.min_track_corners:
                    flow_obj_points = obj_points[flow_ids]
                    ok_pnp, rvec, tvec, pnp_inliers = solve_pose_from_points(
                        flow_obj_points,
                        flow_points,
                        camera_matrix,
                        dist_coeffs,
                        args.pnp_reproj,
                    )

                    if ok_pnp:
                        if last_rvec is not None:
                            angle_deg, trans_dist = pose_jump_metrics(last_rvec, last_tvec, rvec, tvec)
                            if (
                                args.max_angle_jump > 0
                                and angle_deg > args.max_angle_jump
                            ) or (
                                args.max_trans_jump > 0
                                and trans_dist > args.max_trans_jump
                            ):
                                ok_pnp = False
                                jump_reject_count += 1

                    if ok_pnp:
                        if last_rvec is not None and args.pose_smooth > 0:
                            rvec, tvec = smooth_pose(last_rvec, last_tvec, rvec, tvec, args.pose_smooth)

                        last_rvec = rvec.copy()
                        last_tvec = tvec.copy()
                        lost_count = 0
                        tracked_this_frame = True
                        tracked += 1
                        tracked_by_flow += 1

                        tracked_points = flow_points.reshape(-1, 1, 2).astype(np.float32)
                        tracked_ids = flow_ids

                        draw_pose_overlay(frame_aug, box_lower, box_upper, obj_vertices, obj_faces, args.solid, rvec, tvec, camera_matrix, dist_coeffs)

                        for p in flow_points:
                            cv2.circle(frame_aug, tuple(np.int32(p)), 2, (255, 255, 0), -1)

                        status = (
                            f"Tracked(flow) | pts={len(flow_points)} | inliers={pnp_inliers}"
                        )
                        color = (0, 255, 255)

        if not tracked_this_frame:
            lost_count += 1

            if last_rvec is not None and args.hold_frames > 0 and lost_count <= args.hold_frames:
                draw_pose_overlay(
                    frame_aug, box_lower, box_upper, obj_vertices, obj_faces,
                    args.solid, last_rvec, last_tvec, camera_matrix, dist_coeffs,
                )
                if True:
                    status = f"Pose hold {lost_count}/{args.hold_frames}"
                    color = (0, 165, 255)
                    tracked_by_hold += 1
                else:
                    render_reject_count += 1
                    tracked_points = None
                    tracked_ids = None
                    last_rvec = None
                    last_tvec = None
            else:
                tracked_points = None
                tracked_ids = None
                if args.hold_frames == 0 or lost_count > args.hold_frames:
                    last_rvec = None
                    last_tvec = None

        cv2.putText(
            frame_aug,
            status,
            (10, 28),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

        if args.compare_view:
            display = np.hstack([frame, frame_aug])
        else:
            display = frame_aug

        if not args.headless:
            cv2.imshow("Pose Estimation (Chessboard)", display)
        if writer is not None:
            writer.write(display)

        key = cv2.waitKey(1) & 0xFF if not args.headless else -1
        if key == ord("q"):
            break

        prev_gray = gray

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    if processed > 0:
        ratio = 100.0 * tracked / processed
        print(f"Tracking summary: {tracked}/{processed} frames ({ratio:.1f}%)")
        print(
            f"  corners={tracked_by_corners}, flow={tracked_by_flow}, hold={tracked_by_hold}"
        )
        if jump_reject_count > 0:
            print(f"  jump_reject={jump_reject_count}")
        if render_reject_count > 0:
            print(f"  render_reject={render_reject_count}")


if __name__ == "__main__":
    main()

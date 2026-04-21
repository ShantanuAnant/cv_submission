#!/usr/bin/env python3
"""
Manual-Drive Lane Keep Assist  (CLRNet + Frenet + Stanley)
===========================================================

You control the speed with W / S.
The car steers itself using CLRNet lane detection → Frenet planner → Stanley.

Controls
--------
  W        — throttle (hold to accelerate, release to coast)
  S        — brake
  Q / ESC  — quit
  D        — toggle display overlays

Pipeline
--------
  Camera  ──►  CLRNet  ──►  stabilize_lanes (L1=right, L2=left)
                                  │
                    compute_center_route (IPM → world)
                                  │
                   FrenetOptimalTrajectoryPlanner
                                  │
                      StanleyController  ──►  steer only
                                  │
              W/S keyboard  ──►  throttle / brake
                                  │
                         carla.VehicleControl

Display (OpenCV window)
-----------------------
  Colored lane lines  — CLRNet L1 (green), L2 (blue) with stable IDs
  Dashed yellow line  — pixel-space centerline midpoint between L1 & L2
  Bright green line   — Frenet planned trajectory (camera-projected)
  HUD overlay         — speed, steer, source, detection stats

Fixes Applied
-------------
  1. Full trajectory passed to Stanley (not single lookahead point)
  2. Speed-adaptive lookahead distance
  3. Reduced steer smoothing alpha — less lag, more responsiveness
  4. Camera frame gate logic fixed — no longer skips slow frames
  5. Arc-length centerline: normalized pairing at equal normalized s
  6. Convolution uses mode='valid' — no edge bleed at curve entry
  7. Curvature-aware automatic throttle reduction at turns
  8. Stanley k_gain raised slightly for better curve tracking

Usage
-----
    cd simulator_av
    python tests/test_lane_follow_manual_drive.py [--model llamas] [--spawn-index 0]
"""

import sys
import os
import time
import math
import argparse
import logging
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import cv2
except ImportError:
    print("OpenCV required: pip install opencv-python"); sys.exit(1)

try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("manual_lka")


# =============================================================================
# CONFIGURATION
# =============================================================================

CAMERA_WIDTH   = 1280
CAMERA_HEIGHT  = 720
CAMERA_FOV     = 120           # degrees
CAMERA_X       = 2.5           # m forward from vehicle origin
CAMERA_Z       = 1.6           # m height above ground
CAMERA_PITCH   = -0.0          # degrees

MAX_SPEED_KPH  = 20.0          # hard cap
MAX_SPEED_MPS  = MAX_SPEED_KPH / 3.6
LOOKAHEAD_M    = 60.0

# Throttle / brake ramp rates (per frame, ~30 ms/frame)
THROTTLE_RAMP_UP   = 0.06
THROTTLE_RAMP_DOWN = 0.05
BRAKE_RAMP_UP      = 0.12
BRAKE_RAMP_DOWN    = 0.08

# Frenet
FRENET_MAX_ROAD_WIDTH = 3.0
FRENET_OBSTACLE_CLR   = 1.5

# Fallback CARLA waypoints when lanes not detected
FALLBACK_WP_COUNT = 40
FALLBACK_WP_DIST  = 2.0   # m

# FIX 2: Lookahead tuning constants (speed-adaptive)
LOOKAHEAD_MIN_M   = 8.0   # minimum lookahead (m) — was hardcoded 6.0
LOOKAHEAD_SPEED_K = 1.2   # lookahead = max(min, speed_mps * K)

# FIX 3: Steer smoothing — higher alpha = trust fresh output more
STEER_ALPHA = 0.85        # was 0.7 (too laggy)

# FIX 7: Curvature threshold and max throttle in a turn
CURVE_ANGLE_THRESH = 0.15  # radians (~8.5 deg) — triggers throttle limit
CURVE_MAX_THROTTLE = 0.35  # cap throttle when curve detected


# =============================================================================
# CAMERA INTRINSICS
# =============================================================================

def compute_camera_intrinsics(width, height, fov_deg):
    """Return (K, focal_length) for the camera."""
    f = (width / 2.0) / math.tan(math.radians(fov_deg / 2.0))
    K = np.array([[f, 0, width / 2.0],
                  [0, f, height / 2.0],
                  [0, 0, 1.0]], dtype=np.float64)
    return K, f


# =============================================================================
# INVERSE PERSPECTIVE MAPPING  (pixel → vehicle ground plane)
# =============================================================================

def pixel_to_vehicle(u, v, K, cam_height, cam_pitch_deg):
    """
    Project pixel (u, v) onto flat ground and return (x_fwd, y_right)
    in vehicle frame. Returns None if ray misses ground.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    ray_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    p = math.radians(cam_pitch_deg)
    cp, sp = math.cos(p), math.sin(p)
    R = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    ray_veh = R @ ray_cam
    if ray_veh[1] <= 0:
        return None            # ray goes upward → no ground hit
    t = cam_height / ray_veh[1]
    xv = CAMERA_X + ray_veh[2] * t   # forward
    yv = ray_veh[0] * t               # right (positive = right)
    if xv < 0:
        return None
    return xv, yv


# =============================================================================
# WORLD-SPACE ROUTE HELPERS
# =============================================================================

def extract_vehicle_points(lane_points, K, cam_height, cam_pitch_deg):
    """Convert CLRNet lane pixel points to vehicle-frame (xv, yv) sorted by xv."""
    pts = []
    for pt in lane_points:
        r = pixel_to_vehicle(pt.x, pt.y, K, cam_height, cam_pitch_deg)
        if r is None:
            continue
        xv, yv = r
        if 1.0 <= xv <= LOOKAHEAD_M:
            pts.append((xv, yv))
    pts.sort(key=lambda p: p[0])
    return pts


def compute_center_route(right_lane, left_lane, K, cam_height, cam_pitch_deg,
                         ego_x, ego_y, ego_yaw):
    """
    Compute world-space lane centerline.

    FIX 5: Uses normalized arc-length pairing so both lanes are sampled at
    the same *fractional* progress (0→1). This prevents the geometric mismatch
    at curves where the outer lane arc is physically longer than the inner lane,
    which previously caused the centerline to drift toward the inner lane and
    the vehicle to cut corners.

    FIX 6: Smoothing uses np.convolve mode='valid' + explicit xs trimming to
    avoid box-kernel edge bleed at the endpoints (the first few waypoints fed
    to Stanley were previously shifted inward at curve entries).
    """

    HALF_W = 1.75  # half lane width (m)

    r_veh = extract_vehicle_points(
        right_lane.points, K, cam_height, cam_pitch_deg) if right_lane else []
    l_veh = extract_vehicle_points(
        left_lane.points, K, cam_height, cam_pitch_deg) if left_lane else []

    center = []

    def compute_s(points):
        """Cumulative arc-length array for a list of (x, y) points."""
        s = [0.0]
        for i in range(1, len(points)):
            dx = points[i][0] - points[i - 1][0]
            dy = points[i][1] - points[i - 1][1]
            s.append(s[-1] + np.hypot(dx, dy))
        return np.array(s)

    # ------------------------------------------------------------------
    # CASE 1: both lanes visible → normalized arc-length center
    # ------------------------------------------------------------------
    if len(r_veh) >= 4 and len(l_veh) >= 4:
        r_s = compute_s(r_veh)
        l_s = compute_s(l_veh)

        r_x = np.array([p[0] for p in r_veh])
        r_y = np.array([p[1] for p in r_veh])
        l_x = np.array([p[0] for p in l_veh])
        l_y = np.array([p[1] for p in l_veh])

        # FIX 5: normalize both arc-lengths to [0, 1] before pairing
        r_s_norm = r_s / r_s[-1]
        l_s_norm = l_s / l_s[-1]

        for t in np.linspace(0.0, 1.0, 50):
            rx = float(np.interp(t, r_s_norm, r_x))
            ry = float(np.interp(t, r_s_norm, r_y))
            lx = float(np.interp(t, l_s_norm, l_x))
            ly = float(np.interp(t, l_s_norm, l_y))
            center.append(((rx + lx) / 2.0, (ry + ly) / 2.0))

    # ------------------------------------------------------------------
    # CASE 2: only right lane → shift left by half lane
    # ------------------------------------------------------------------
    elif len(r_veh) >= 4:
        center = [(x, y - HALF_W) for x, y in r_veh]

    # ------------------------------------------------------------------
    # CASE 3: only left lane → shift right by half lane
    # ------------------------------------------------------------------
    elif len(l_veh) >= 4:
        center = [(x, y + HALF_W) for x, y in l_veh]

    else:
        return []

    # ------------------------------------------------------------------
    # FIX 6: Smooth lateral (y) positions — mode='valid' trims edges
    # instead of bleeding zeros into the boundary samples.
    # Kernel size 5 → trims 2 points each side; adjust if needed.
    # ------------------------------------------------------------------
    KERNEL = 5
    TRIM   = KERNEL // 2   # = 2

    if len(center) >= KERNEL + 2:
        xs = np.array([p[0] for p in center])
        ys = np.array([p[1] for p in center])
        kernel = np.ones(KERNEL) / KERNEL
        ys_smooth = np.convolve(ys, kernel, mode='valid')   # length = N - 4
        xs_trim   = xs[TRIM:-TRIM]                          # same length
        center    = list(zip(xs_trim, ys_smooth))

    # ------------------------------------------------------------------
    # Vehicle frame → world frame
    # ------------------------------------------------------------------
    cos_y = math.cos(ego_yaw)
    sin_y = math.sin(ego_yaw)

    world = []
    for xv, yv in center:
        wx = ego_x + xv * cos_y - yv * sin_y
        wy = ego_y + xv * sin_y + yv * cos_y
        world.append((wx, wy, xv))

    world.sort(key=lambda p: p[2])
    return [(p[0], p[1]) for p in world]


# =============================================================================
# FIX 7: CURVATURE ESTIMATOR
# =============================================================================

def estimate_turn_angle(world_wp):
    """
    Return the maximum heading change (radians) across the first few waypoints.
    Used to detect an upcoming curve and cap throttle proactively.
    """
    if len(world_wp) < 3:
        return 0.0
    max_angle = 0.0
    for i in range(1, min(len(world_wp) - 1, 6)):
        dx1 = world_wp[i][0]     - world_wp[i - 1][0]
        dy1 = world_wp[i][1]     - world_wp[i - 1][1]
        dx2 = world_wp[i + 1][0] - world_wp[i][0]
        dy2 = world_wp[i + 1][1] - world_wp[i][1]
        a1  = math.atan2(dy1, dx1)
        a2  = math.atan2(dy2, dx2)
        diff = abs(a2 - a1)
        if diff > math.pi:
            diff = 2 * math.pi - diff
        max_angle = max(max_angle, diff)
    return max_angle


# =============================================================================
# CARLA WAYPOINT FALLBACK
# =============================================================================

def get_carla_waypoints_ahead(carla_map, ex, ey, ez, count, spacing):
    loc = carla.Location(x=ex, y=ey, z=ez)
    wp  = carla_map.get_waypoint(loc, project_to_road=True,
                                 lane_type=carla.LaneType.Driving)
    if wp is None:
        return []
    pts = [(wp.transform.location.x, wp.transform.location.y)]
    for _ in range(count - 1):
        nxt = wp.next(spacing)
        if not nxt:
            break
        wp = nxt[0]
        pts.append((wp.transform.location.x, wp.transform.location.y))
    return pts


# =============================================================================
# CV DRAWING UTILITIES
# =============================================================================

def _world_to_pixel(wx, wy, ego_x, ego_y, ego_yaw, K,
                    cam_x=CAMERA_X, cam_z=CAMERA_Z, cam_pitch=CAMERA_PITCH):
    """Project world (wx, wy) to image pixel, or None if behind/off-image."""
    dx, dy = wx - ego_x, wy - ego_y
    ch, sh = math.cos(-ego_yaw), math.sin(-ego_yaw)
    xv = dx * ch - dy * sh
    yv = dx * sh + dy * ch
    if xv < 0.3:
        return None
    xv -= cam_x
    p  = math.radians(cam_pitch)
    cp, sp = math.cos(p), math.sin(p)
    xc = yv
    yc = -cam_z * cp + xv * sp
    zc =  cam_z * sp + xv * cp
    if zc <= 0:
        return None
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    return int(fx * xc / zc + cx), int(fy * yc / zc + cy)


def draw_trajectory_on_frame(frame, traj_pts, K, ego_x, ego_y, ego_yaw,
                              color=(0, 255, 80), thickness=3):
    """Overlay Frenet planned trajectory as a solid polyline."""
    h, w = frame.shape[:2]
    pixels = []
    for pt in traj_pts:
        px = _world_to_pixel(pt.x, pt.y, ego_x, ego_y, ego_yaw, K)
        if px:
            u, v = px
            if 0 <= u < w and 0 <= v < h:
                pixels.append((u, v))
    for i in range(len(pixels) - 1):
        cv2.line(frame, pixels[i], pixels[i + 1], color, thickness, cv2.LINE_AA)


def _dashed_polyline(img, pts, dash_px=18, gap_px=12,
                     color=(0, 215, 255), thickness=2):
    """Draw a dashed polyline through `pts` (list of (x,y) integers)."""
    if len(pts) < 2:
        return
    drawing = True
    bucket  = 0.0
    target  = float(dash_px)

    for i in range(len(pts) - 1):
        x1, y1 = float(pts[i][0]),   float(pts[i][1])
        x2, y2 = float(pts[i+1][0]), float(pts[i+1][1])
        seg = math.hypot(x2 - x1, y2 - y1)
        if seg < 0.5:
            continue
        dx, dy    = (x2 - x1) / seg, (y2 - y1) / seg
        remaining = seg
        fx, fy    = x1, y1

        while remaining > 1e-3:
            step = min(remaining, target - bucket)
            nx, ny = fx + dx * step, fy + dy * step
            if drawing:
                cv2.line(img,
                         (int(round(fx)), int(round(fy))),
                         (int(round(nx)), int(round(ny))),
                         color, thickness, cv2.LINE_AA)
            bucket    += step
            remaining -= step
            fx, fy     = nx, ny
            if bucket >= target - 1e-6:
                bucket  = 0.0
                drawing = not drawing
                target  = float(gap_px) if not drawing else float(dash_px)


def draw_pixel_centerline(frame, right_lane, left_lane,
                          dash_px=22, gap_px=14,
                          color=(0, 215, 255), thickness=2):
    """Draw a dashed centerline between L1 and L2 in pixel space."""
    if right_lane is None or left_lane is None:
        return
    if len(right_lane.points) < 3 or len(left_lane.points) < 3:
        return

    r_pts_raw = sorted([(p.x, p.y) for p in right_lane.points], key=lambda p: p[1])
    l_pts_raw = sorted([(p.x, p.y) for p in left_lane.points],  key=lambda p: p[1])

    r_y = np.array([p[1] for p in r_pts_raw], dtype=np.float32)
    r_x = np.array([p[0] for p in r_pts_raw], dtype=np.float32)
    l_y = np.array([p[1] for p in l_pts_raw], dtype=np.float32)
    l_x = np.array([p[0] for p in l_pts_raw], dtype=np.float32)

    y_lo = max(r_y[0],  l_y[0])
    y_hi = min(r_y[-1], l_y[-1])
    if y_lo >= y_hi:
        return

    y_samp = np.arange(y_lo, y_hi, 4.0)
    center_pts = []
    for y in y_samp:
        if (r_y[0] <= y <= r_y[-1]) and (l_y[0] <= y <= l_y[-1]):
            rx = float(np.interp(y, r_y, r_x))
            lx = float(np.interp(y, l_y, l_x))
            cx = int((rx + lx) / 2.0)
            center_pts.append((cx, int(y)))

    _dashed_polyline(frame, center_pts, dash_px, gap_px, color, thickness)


def draw_throttle_brake_bars(frame, throttle, brake, speed_kph, max_kph,
                              w=CAMERA_WIDTH, h=CAMERA_HEIGHT):
    """Compact HUD bar in the bottom-right corner."""
    bar_w, bar_h = 160, 16
    x0 = w - bar_w - 20
    for label, val, y_off, col in [
        ("THR", throttle, 90, (0, 200, 80)),
        ("BRK", brake,    65, (0,  60, 220)),
    ]:
        y = h - y_off
        cv2.putText(frame, label, (x0 - 40, y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x0, y), (x0 + bar_w, y + bar_h), (60, 60, 60), -1)
        cv2.rectangle(frame, (x0, y),
                      (x0 + int(bar_w * val), y + bar_h), col, -1)
    # Speed
    y = h - 40
    cv2.putText(frame, "SPD", (x0 - 40, y + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (x0, y), (x0 + bar_w, y + bar_h), (60, 60, 60), -1)
    frac = min(1.0, speed_kph / max_kph)
    col  = (0, 255, 0) if frac < 0.7 else (0, 165, 255) if frac < 0.9 else (0, 0, 255)
    cv2.rectangle(frame, (x0, y), (x0 + int(bar_w * frac), y + bar_h), col, -1)


# =============================================================================
# CARLA CLEANUP HELPER
# =============================================================================

def cleanup_carla():
    try:
        c = carla.Client("localhost", 2000)
        c.set_timeout(8.0)
        w = c.get_world()
        for v in w.get_actors().filter("vehicle.*"):
            v.destroy()
        for s in w.get_actors().filter("sensor.*"):
            s.destroy()
        time.sleep(0.4)
    except Exception:
        pass


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Manual-Drive Lane Keep Assist (CLRNet + Frenet + Stanley)")
    parser.add_argument("--model", default="llamas",
                        choices=["llamas", "tusimple", "culane"])
    parser.add_argument("--spawn-index", type=int, default=0)
    parser.add_argument("--duration", type=int, default=0,
                        help="Max duration seconds (0 = unlimited)")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Lazy imports
    # ------------------------------------------------------------------
    from nodes.carla_bridge_node import CarlaBridgeNode, CarlaBridgeConfig
    from nodes.planning_node import PlanningNode, PlanningConfig, EgoState
    from nodes.control_node import ControlNode, ControlConfig
    from planning import FrenetOptimalTrajectoryPlanner
    from control import StanleyController, TrajectoryPoint as CtrlTrajPoint
    from carla_bridge import SensorManager, SensorConfig, SensorType
    from perception.clrnet_detector import CLRNetDetector, CLRNET_AVAILABLE
    from perception.lane_stabilizer import (
        stabilize_lanes, draw_stabilized_lanes, get_lane_by_id
    )

    if not CLRNET_AVAILABLE:
        logger.error("CLRNet not available. Check installation.")
        return False

    logger.info("=" * 70)
    logger.info("MANUAL-DRIVE LANE KEEP ASSIST  (CLRNet + Frenet + Stanley)")
    logger.info("=" * 70)
    logger.info("  W = throttle   S = brake   Q/ESC = quit   D = toggle overlay")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # 1. Connect to CARLA
    # ------------------------------------------------------------------
    cleanup_carla()
    logger.info("\n[1/6] Connecting to CARLA...")

    bridge_cfg = CarlaBridgeConfig(
        host="localhost", port=2000,
        vehicle_blueprint="vehicle.jeep.wrangler_rubicon",
        sync_mode=False,
        fixed_delta_seconds=0.05,
        enable_sensors=False
    )
    bridge = CarlaBridgeNode(bridge_cfg)
    if not bridge.connect():
        logger.error("Failed to connect to CARLA"); return False
    logger.info("✓ Connected to CARLA")

    carla_world = bridge._client.get_world()
    carla_world.set_weather(carla.WeatherParameters.ClearSunset)
    carla_map   = carla_world.get_map()

    # ------------------------------------------------------------------
    # 2. Spawn vehicle
    # ------------------------------------------------------------------
    logger.info("\n[2/6] Spawning vehicle...")
    if not bridge.spawn_vehicle(spawn_index=args.spawn_index):
        logger.error("Failed to spawn vehicle"); bridge.disconnect(); return False

    bridge.start()
    time.sleep(0.4)

    t = bridge.get_vehicle_transform()
    if not t:
        logger.error("Cannot read vehicle transform"); bridge.disconnect(); return False
    loc, rot = t
    spawn_x, spawn_y, spawn_z = loc
    logger.info(f"✓ Vehicle at ({spawn_x:.1f}, {spawn_y:.1f}), heading {rot[1]:.1f}°")

    # ------------------------------------------------------------------
    # 3. Camera + CLRNet
    # ------------------------------------------------------------------
    logger.info("\n[3/6] Setting up camera & CLRNet...")

    sensors = SensorManager(
        bridge._client, bridge._ego_vehicle,
        SensorConfig(camera_width=CAMERA_WIDTH,
                     camera_height=CAMERA_HEIGHT,
                     camera_fov=CAMERA_FOV)
    )
    cam_tf = carla.Transform(
        carla.Location(x=CAMERA_X, y=0.0, z=CAMERA_Z),
        carla.Rotation(pitch=CAMERA_PITCH, yaw=0.0, roll=0.0)
    )
    sensors.add_camera(SensorType.CAMERA_RGB, cam_tf,
                       CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FOV)
    time.sleep(0.6)

    detector = CLRNetDetector(model_type=args.model, device="cuda",
                              conf_threshold=0.4)
    if not detector.is_ready:
        logger.error("CLRNet failed to load"); return False
    logger.info("✓ CLRNet ready")

    K, _ = compute_camera_intrinsics(CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FOV)

    # ------------------------------------------------------------------
    # 4. Frenet Planner
    # ------------------------------------------------------------------
    logger.info("\n[4/6] Setting up Frenet planner...")

    planning_node = PlanningNode(
        PlanningConfig(rate_hz=20.0, default_planner="frenet_optimal"))
    frenet = FrenetOptimalTrajectoryPlanner(
        max_speed=MAX_SPEED_MPS,
        max_road_width=FRENET_MAX_ROAD_WIDTH,
        obstacle_clearance=FRENET_OBSTACLE_CLR
    )
    planning_node.register_planner(frenet)
    planning_node.set_active_planner("frenet_optimal")
    planning_node.start()
    logger.info("✓ Frenet planner ready")

    # ------------------------------------------------------------------
    # 5. Stanley Controller
    #    FIX 8: k_gain raised to 1.0 (was 0.8) for better curve tracking.
    #    Lateral-only: speed_kp / speed_ki remain 0.
    # ------------------------------------------------------------------
    logger.info("\n[5/6] Setting up Stanley controller (lateral only)...")

    control_node = ControlNode(ControlConfig(
        rate_hz=50.0,
        default_controller="stanley",
        target_speed=0.0
    ))
    stanley = StanleyController(
        k_gain=1.0,             # FIX 8: was 0.8
        k_soft=1.0,
        max_steer=math.radians(30),
        speed_kp=0.0,
        speed_ki=0.0
    )
    control_node.register_controller(stanley)
    control_node.set_controller("stanley")
    control_node.start()
    logger.info("✓ Stanley controller ready (LATERAL ONLY)")

    # Warm-up
    time.sleep(0.4)
    t = bridge.get_vehicle_transform()
    if t:
        loc, rot = t
        planning_node._ego_state = EgoState(
            x=loc[0], y=loc[1], z=loc[2],
            heading=math.radians(rot[1]),
            speed=0.0, timestamp=time.time()
        )

    # ------------------------------------------------------------------
    # 6. Display window
    # ------------------------------------------------------------------
    if not args.no_display:
        cv2.namedWindow("Lane Keep Assist", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Lane Keep Assist", CAMERA_WIDTH, CAMERA_HEIGHT)

    # ------------------------------------------------------------------
    # 7. Main loop
    # ------------------------------------------------------------------
    logger.info("\n[6/6] Running manual-drive lane keep assist...")
    logger.info("=" * 70)
    logger.info("  W = throttle   S = brake   Q/ESC = quit   D = toggle overlay")
    logger.info("=" * 70)

    start_time   = time.time()
    last_log     = start_time
    frame_count  = 0
    lane_used    = 0
    fallback_cnt = 0

    throttle     = 0.0
    brake        = 0.0
    steer_out    = 0.0
    prev_steer   = 0.0
    show_overlay = True

    # FIX 4: correct frame-rate gate — track last frame timestamp
    last_frame_time = 0.0
    FRAME_MIN_INTERVAL = 1.0 / 35.0  # ~35 fps cap; skip only if too fast

    try:
        while True:
            elapsed = time.time() - start_time
            if args.duration > 0 and elapsed > args.duration:
                logger.info(f"Duration limit reached ({args.duration}s)")
                break

            # ---- Keyboard input (non-blocking; 20 ms) --------------------
            key = cv2.waitKey(20) & 0xFF if not args.no_display else 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("d"):
                show_overlay = not show_overlay

            # ---- Ego state -----------------------------------------------
            t_raw = bridge.get_vehicle_transform()
            if not t_raw:
                time.sleep(0.01)
                continue
            loc, rot = t_raw
            ego_x, ego_y, ego_z = loc
            ego_yaw   = math.radians(rot[1])
            ego_speed = bridge.get_vehicle_speed()

            # ---- Throttle / brake ramp -----------------------------------
            if key == ord("w"):
                if ego_speed < MAX_SPEED_MPS:
                    throttle = min(1.0, throttle + THROTTLE_RAMP_UP)
                else:
                    throttle = max(0.0, throttle - THROTTLE_RAMP_DOWN)
                brake = max(0.0, brake - BRAKE_RAMP_DOWN * 2)
            elif key == ord("s"):
                brake    = min(1.0, brake    + BRAKE_RAMP_UP)
                throttle = max(0.0, throttle - THROTTLE_RAMP_DOWN * 2)
            else:
                throttle = max(0.0, throttle - THROTTLE_RAMP_DOWN)
                brake    = max(0.0, brake    - BRAKE_RAMP_DOWN)

            planning_node._ego_state = EgoState(
                x=ego_x, y=ego_y, z=ego_z,
                heading=ego_yaw, speed=ego_speed,
                timestamp=time.time()
            )

            # ---- Camera frame --------------------------------------------
            # FIX 4: only skip if arriving *too fast* (cap at ~35 fps).
            #         Never skip slow frames — those carry the curve data.
            now = time.time()
            if now - last_frame_time < FRAME_MIN_INTERVAL:
                # Frame arrived too quickly; skip to avoid overloading GPU.
                # Still apply the control computed last iteration.
                bridge._ego_vehicle.apply_control(carla.VehicleControl(
                    throttle=float(throttle),
                    brake=float(brake),
                    steer=float(steer_out),
                    hand_brake=False,
                    reverse=False
                ))
                continue
            last_frame_time = now

            cam_data = sensors.get_camera_image()
            if cam_data is None or cam_data.image is None:
                time.sleep(0.01)
                continue

            frame = cam_data.image
            if frame.shape[2] == 4:
                frame = frame[:, :, :3].copy()
            else:
                frame = frame.copy()

            # ---- CLRNet detection & stabilization -----------------------
            lane_result = detector.detect(frame, draw_lanes=False)
            lane_result = stabilize_lanes(lane_result, CAMERA_WIDTH, CAMERA_HEIGHT)
            frame_count += 1

            right_lane = get_lane_by_id(lane_result, 1)   # L1 — right boundary
            left_lane  = get_lane_by_id(lane_result, 2)   # L2 — left  boundary

            # ---- World-space centerline route ---------------------------
            route    = None
            world_wp = compute_center_route(
                right_lane, left_lane, K, CAMERA_Z, CAMERA_PITCH,
                ego_x, ego_y, ego_yaw
            )

            if world_wp and len(world_wp) >= 4:
                route = world_wp
                lane_used += 1

                # FIX 7: curvature-aware throttle cap
                turn_angle = estimate_turn_angle(world_wp)
                if turn_angle > CURVE_ANGLE_THRESH:
                    throttle = min(throttle, CURVE_MAX_THROTTLE)

            # ---- Frenet planning ----------------------------------------
            if route and len(route) > 1:
                planning_node.set_route(route)

            planning_node.process_frame()

            # ---- FIX 1 + FIX 2: full trajectory to Stanley,
            #      speed-adaptive lookahead
            # --------------------------------------------------------------
            if (planning_node._current_trajectory and
                    planning_node._current_trajectory.is_valid):

                traj = planning_node._current_trajectory.points

                # FIX 2: adaptive lookahead — scales with speed
                lookahead_dist = max(LOOKAHEAD_MIN_M,
                                     ego_speed * LOOKAHEAD_SPEED_K)

                # Find the index of the first point beyond lookahead_dist
                lookahead_idx = len(traj) - 1
                for i, pt in enumerate(traj):
                    dist = math.hypot(pt.x - ego_x, pt.y - ego_y)
                    if dist > lookahead_dist:
                        lookahead_idx = i
                        break

                # FIX 1: pass the full trajectory (from ego up to lookahead)
                # so Stanley has the complete path geometry for CTE + heading.
                control_node._trajectory = [
                    CtrlTrajPoint(
                        x=pt.x, y=pt.y, z=pt.z,
                        velocity=pt.velocity,
                        heading=pt.heading,
                        curvature=pt.curvature,
                        time_offset=pt.time_offset
                    )
                    for pt in traj[:lookahead_idx + 1]
                ]
                control_node._trajectory_timestamp = time.time()

            # ---- Stanley → LATERAL steer only ---------------------------
            control_node._vehicle_state.x       = ego_x
            control_node._vehicle_state.y        = ego_y
            control_node._vehicle_state.heading  = ego_yaw
            control_node._vehicle_state.speed    = ego_speed

            ctrl = control_node.compute_control()

            # FIX 3: higher alpha = fresher steer, less lag
            raw_steer = ctrl.steer
            steer_out = STEER_ALPHA * raw_steer + (1.0 - STEER_ALPHA) * prev_steer
            prev_steer = steer_out
            steer_out  = float(np.clip(steer_out, -0.5, 0.5))

            # ---- Apply control ------------------------------------------
            bridge._ego_vehicle.apply_control(carla.VehicleControl(
                throttle=float(throttle),
                brake=float(brake),
                steer=steer_out,
                hand_brake=False,
                reverse=False
            ))

            if bridge.config.sync_mode:
                bridge.tick()
            bridge.publish_vehicle_state()

            # ---- Display ------------------------------------------------
            if not args.no_display:
                display = draw_stabilized_lanes(frame, lane_result, draw_ids=True)

                if show_overlay:
                    draw_pixel_centerline(
                        display, right_lane, left_lane,
                        dash_px=22, gap_px=14,
                        color=(0, 215, 255), thickness=2
                    )

                    if (planning_node._current_trajectory and
                            planning_node._current_trajectory.is_valid):
                        draw_trajectory_on_frame(
                            display,
                            planning_node._current_trajectory.points,
                            K, ego_x, ego_y, ego_yaw,
                            color=(0, 255, 80), thickness=3
                        )

                speed_kph = ego_speed * 3.6
                source    = ("LANE" if (right_lane or left_lane) and
                             lane_used > fallback_cnt else "WAYPOINT")
                ids_str   = ", ".join(f"L{l.lane_id}" for l in lane_result.lanes)
                steer_deg = math.degrees(steer_out * math.radians(30))

                # Adaptive lookahead shown in HUD for tuning visibility
                lookahead_disp = max(LOOKAHEAD_MIN_M,
                                     ego_speed * LOOKAHEAD_SPEED_K)

                hud = [
                    f"Source    : {source}",
                    f"Speed     : {speed_kph:5.1f} km/h",
                    f"Steer     : {steer_deg:+5.1f} deg  ({steer_out:+.3f})",
                    f"Lookahead : {lookahead_disp:.1f} m",
                    f"THR: {'[W]' if throttle > 0.01 else '   '}  "
                    f"BRK: {'[S]' if brake > 0.01 else '   '}",
                    f"Lanes     : {lane_result.num_lanes} [{ids_str}]",
                    f"Frame     : {frame_count}   {elapsed:.0f}s",
                ]
                y = 30
                for line in hud:
                    cv2.putText(display, line, (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                                (0, 255, 0), 2, cv2.LINE_AA)
                    y += 28

                draw_throttle_brake_bars(display, throttle, brake,
                                         speed_kph, MAX_SPEED_KPH)

                cv2.putText(display, "DASHED YELLOW = lane centerline",
                            (10, CAMERA_HEIGHT - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 215, 255), 1, cv2.LINE_AA)
                cv2.putText(display, "GREEN = Frenet trajectory",
                            (10, CAMERA_HEIGHT - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 80), 1, cv2.LINE_AA)

                cv2.imshow("Lane Keep Assist", display)

            # ---- Periodic log -------------------------------------------
            if time.time() - last_log >= 3.0:
                pct = 100.0 * lane_used / max(1, frame_count)
                logger.info(
                    f"[{elapsed:.0f}s]  Speed: {ego_speed*3.6:.1f} km/h  "
                    f"Steer: {steer_out:+.3f}  "
                    f"Lane-guided: {pct:.0f}%  "
                    f"Pos: ({ego_x:.1f}, {ego_y:.1f})"
                )
                last_log = time.time()

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    logger.info("\n[CLEANUP] Stopping...")
    cv2.destroyAllWindows()
    detector.stop()

    try:
        bridge._ego_vehicle.apply_control(
            carla.VehicleControl(throttle=0.0, brake=1.0))
    except Exception:
        pass
    time.sleep(0.4)

    control_node.stop()
    planning_node.stop()
    sensors.destroy_all()
    bridge.stop()

    elapsed  = time.time() - start_time
    lane_pct = 100.0 * lane_used / max(1, frame_count)
    logger.info("\n" + "=" * 70)
    logger.info("STATISTICS")
    logger.info("=" * 70)
    logger.info(f"Duration        : {elapsed:.1f}s")
    logger.info(f"Total frames    : {frame_count}")
    logger.info(f"Lane-guided     : {lane_used}  ({lane_pct:.1f}%)")
    logger.info(f"Fallback frames : {fallback_cnt}")
    logger.info("=" * 70)
    logger.info("✓ Done!")
    return True


if __name__ == "__main__":
    try:
        ok = main()
        sys.exit(0 if ok else 1)
    except Exception as exc:
        logger.error(f"Test failed: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
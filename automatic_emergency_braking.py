#!/usr/bin/env python3
"""
AEB (Autonomous Emergency Braking) Test
========================================

Simple test scenario:
  • Ego vehicle — manually driven via WASD, speed-limited to 30 km/h.
  • Stationary target vehicle — parked 50 m ahead in the same lane.
  • YOLO detects the target; CARLA native depth gives the distance.
  • AEB monitors Time-To-Collision (TTC) and overrides with full braking
    when TTC drops below the threshold.

Controls:
    W — throttle        A — steer left
    S — brake           D — steer right
    Q / ESC — quit

Usage:
    cd simulator_av
    python -m tests.aeb.aeb_test
    # or:
    cd tests/aeb && python aeb_test.py
"""

import sys
import os
import time
import math
import threading
import logging

# Project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("aeb_test")

import cv2
import numpy as np
from ultralytics import YOLO

# Reuse perception helpers from the depth-detection module
from perception.object_detection.carla_stereo_depth import (
    carla_depth_to_meters,
    get_distance_at_bbox,
    YOLO_MODEL,
    YOLO_CONF,
    YOLO_IMGSZ,
    ALLOWED_CLASSES,
    CLASS_COLORS,
    DEFAULT_COLOR,
    MAX_DEPTH_M,
    MIN_DEPTH_M,
)

from carla_bridge import CarlaClient, VehicleSpawner, VehicleSpawnerConfig

try:
    import carla
except ImportError:
    logger.error("CARLA Python API not available")
    sys.exit(1)


# ============================================================================
# PARAMETERS
# ============================================================================

# Camera
CAMERA_W, CAMERA_H, CAMERA_FOV = 1280, 720, 90.0
MOUNT_X, MOUNT_Y, MOUNT_Z, MOUNT_PITCH = 0.5, 0.0, 2.3, -5.0

# Driving limits
MAX_SPEED_KPH   = 30.0          # hard speed cap
MAX_SPEED_MPS   = MAX_SPEED_KPH / 3.6

# AEB thresholds
TTC_WARN_S      = 3.0           # yellow warning
TTC_BRAKE_S     = 1.5           # hard emergency brake
AEB_DECEL       = 8.0           # m/s²  (full braking deceleration target)

# Target vehicle distance
TARGET_DISTANCE_M = 30.0        # metres ahead of ego in the same lane

# Smooth driving feel  (racing-game style ramp rates per frame)
THROTTLE_RAMP_UP   = 0.08       # how fast throttle builds when W held
THROTTLE_RAMP_DOWN = 0.06       # how fast throttle fades on release
BRAKE_RAMP_UP      = 0.15       # brakes respond faster than throttle
BRAKE_RAMP_DOWN    = 0.10
STEER_RAMP_UP      = 0.04       # steering rate while A/D held
STEER_CENTRE_RATE  = 0.08       # auto-centre speed on release


# ============================================================================
# HELPERS
# ============================================================================

def get_speed_mps(vehicle) -> float:
    """Return vehicle speed in m/s."""
    v = vehicle.get_velocity()
    return math.sqrt(v.x**2 + v.y**2 + v.z**2)


def compute_ttc(distance_m: float, speed_mps: float) -> float:
    """
    Time-to-collision for approaching a stationary object.
    Returns inf if speed is ~0 or distance is invalid.
    """
    if speed_mps < 0.1 or distance_m is None or distance_m <= 0:
        return float("inf")
    return distance_m / speed_mps


def spawn_target_vehicle(world, ego, distance_m: float):
    """
    Spawn a stationary vehicle *distance_m* metres in front of the ego,
    aligned with the ego's heading.
    """
    ego_tf = ego.get_transform()
    yaw_rad = math.radians(ego_tf.rotation.yaw)

    target_loc = carla.Location(
        x=ego_tf.location.x + distance_m * math.cos(yaw_rad),
        y=ego_tf.location.y + distance_m * math.sin(yaw_rad),
        z=ego_tf.location.z,
    )
    target_rot = carla.Rotation(pitch=0, yaw=ego_tf.rotation.yaw, roll=0)
    target_tf  = carla.Transform(target_loc, target_rot)

    bp_lib = world.get_blueprint_library()
    # Pick a large, visible vehicle
    bp = bp_lib.find("vehicle.ford.mustang")
    if bp.has_attribute("color"):
        bp.set_attribute("color", "255,0,0")  # bright red

    target = world.try_spawn_actor(bp, target_tf)
    if target is None:
        # Snap to nearest waypoint and retry
        waypoint = world.get_map().get_waypoint(target_loc,
                                                 project_to_road=True,
                                                 lane_type=carla.LaneType.Driving)
        if waypoint is not None:
            target_tf = waypoint.transform
            target_tf.location.z += 0.5
            target = world.try_spawn_actor(bp, target_tf)

    if target is not None:
        # Engage handbrake so it doesn't roll
        target.apply_control(carla.VehicleControl(hand_brake=True))
        logger.info(f"Target vehicle spawned {distance_m:.0f}m ahead at "
                    f"({target_tf.location.x:.1f}, {target_tf.location.y:.1f})")
    else:
        logger.warning("Could not spawn target vehicle — test will still run")

    return target


def spawn_target_pedestrian(world, ego, distance_m: float):
    """
    Spawn a stationary pedestrian *distance_m* metres in front of the ego,
    standing in the middle of the lane.
    """
    ego_tf = ego.get_transform()
    yaw_rad = math.radians(ego_tf.rotation.yaw)

    target_loc = carla.Location(
        x=ego_tf.location.x + distance_m * math.cos(yaw_rad),
        y=ego_tf.location.y + distance_m * math.sin(yaw_rad),
        z=ego_tf.location.z + 1.0,   # slightly above ground so it doesn't clip
    )
    target_rot = carla.Rotation(pitch=0, yaw=ego_tf.rotation.yaw - 180, roll=0)
    target_tf  = carla.Transform(target_loc, target_rot)

    bp_lib = world.get_blueprint_library()
    walker_bps = bp_lib.filter("walker.pedestrian.*")
    if not walker_bps:
        logger.warning("No pedestrian blueprints found")
        return None

    import random
    bp = random.choice(walker_bps)
    if bp.has_attribute("is_invincible"):
        bp.set_attribute("is_invincible", "false")

    ped = world.try_spawn_actor(bp, target_tf)
    if ped is None:
        # Try snapping to road
        waypoint = world.get_map().get_waypoint(target_loc,
                                                 project_to_road=True,
                                                 lane_type=carla.LaneType.Driving)
        if waypoint is not None:
            target_tf = waypoint.transform
            target_tf.location.z += 1.0
            ped = world.try_spawn_actor(bp, target_tf)

    if ped is not None:
        logger.info(f"Target pedestrian spawned {distance_m:.0f}m ahead at "
                    f"({target_tf.location.x:.1f}, {target_tf.location.y:.1f})")
    else:
        logger.warning("Could not spawn target pedestrian — test will still run")

    return ped


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="AEB Test")
    parser.add_argument("--target", type=str, default="vehicle",
                        choices=["vehicle", "pedestrian"],
                        help="Type of obstacle to spawn ahead")
    parser.add_argument("--distance", type=float, default=TARGET_DISTANCE_M,
                        help="Distance to spawn the target (meters)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1.  Connect to CARLA
    # ------------------------------------------------------------------
    client = CarlaClient()
    if not client.connect():
        logger.error("Cannot connect to CARLA")
        return

    carla_client = client._client
    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05   # 20 Hz
    world.apply_settings(settings)

    # Traffic manager (needed for spawner internals, but we drive manually)
    tm = carla_client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)

    # ------------------------------------------------------------------
    # 2.  Spawn ego vehicle  (manual control)
    # ------------------------------------------------------------------
    spawner = VehicleSpawner(client)
    # spawner._config = VehicleSpawnerConfig(default_ego_blueprint="vehicle.dodge.charger")
    ego = spawner.spawn_ego(spawn_index=1)
    if ego is None:
        logger.error("Failed to spawn ego")
        return
    logger.info("Ego vehicle spawned — manual WASD control")

    # Let physics settle
    for _ in range(5):
        world.tick()
        time.sleep(0.05)

    # ------------------------------------------------------------------
    # 3.  Spawn stationary target ahead
    # ------------------------------------------------------------------
    if args.target == "pedestrian":
        target = spawn_target_pedestrian(world, ego, args.distance)
    else:
        target = spawn_target_vehicle(world, ego, args.distance)

    # ------------------------------------------------------------------
    # 4.  Cameras  (RGB + Depth, co-located)
    # ------------------------------------------------------------------
    bp_lib = world.get_blueprint_library()
    cam_tf = carla.Transform(
        carla.Location(x=MOUNT_X, y=MOUNT_Y, z=MOUNT_Z),
        carla.Rotation(pitch=MOUNT_PITCH),
    )

    rgb_bp = bp_lib.find("sensor.camera.rgb")
    rgb_bp.set_attribute("image_size_x", str(CAMERA_W))
    rgb_bp.set_attribute("image_size_y", str(CAMERA_H))
    rgb_bp.set_attribute("fov", str(CAMERA_FOV))
    rgb_cam = world.spawn_actor(rgb_bp, cam_tf, attach_to=ego)

    depth_bp = bp_lib.find("sensor.camera.depth")
    depth_bp.set_attribute("image_size_x", str(CAMERA_W))
    depth_bp.set_attribute("image_size_y", str(CAMERA_H))
    depth_bp.set_attribute("fov", str(CAMERA_FOV))
    depth_cam = world.spawn_actor(depth_bp, cam_tf, attach_to=ego)

    latest = {"rgb": None, "depth": None}
    cb_lock = threading.Lock()

    def _on_rgb(image):
        a = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
            (image.height, image.width, 4))
        with cb_lock:
            latest["rgb"] = a.copy()

    def _on_depth(image):
        a = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
            (image.height, image.width, 4))
        with cb_lock:
            latest["depth"] = a.copy()

    rgb_cam.listen(_on_rgb)
    depth_cam.listen(_on_depth)

    # ------------------------------------------------------------------
    # 5.  YOLO
    # ------------------------------------------------------------------
    model = YOLO(YOLO_MODEL, task="detect")
    logger.info(f"YOLO model loaded: {YOLO_MODEL}")

    # ------------------------------------------------------------------
    # 6.  Warm-up
    # ------------------------------------------------------------------
    for _ in range(15):
        world.tick()
        time.sleep(0.05)
    logger.info("Warm-up done — drive with WASD.  AEB active.")

    # ------------------------------------------------------------------
    # 7.  Main loop
    # ------------------------------------------------------------------
    cv2.namedWindow("AEB TEST", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AEB TEST", CAMERA_W, CAMERA_H)

    steer = 0.0
    throttle = 0.0
    brake = 0.0
    aeb_active = False
    fps = 0.0
    frame_count = 0
    last_fps_time = time.time()

    try:
        while True:
            t0 = time.time()
            world.tick()
            time.sleep(0.01)

            # ---- Grab frames --------------------------------------------
            with cb_lock:
                rgb_raw   = latest["rgb"]
                depth_raw = latest["depth"]
            if rgb_raw is None or depth_raw is None:
                continue

            frame  = np.ascontiguousarray(rgb_raw[:, :, :3])
            depth_m = carla_depth_to_meters(depth_raw)

            # ---- YOLO detection -----------------------------------------
            results = model(frame, verbose=False, conf=YOLO_CONF,
                            imgsz=YOLO_IMGSZ, device="cuda")[0]

            closest_dist = None
            for box in results.boxes:
                cls = int(box.cls[0])
                if cls not in ALLOWED_CLASSES:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf  = float(box.conf[0])
                label = model.names[cls]
                dist  = get_distance_at_bbox(depth_m, x1, y1, x2, y2)

                color = CLASS_COLORS.get(cls, DEFAULT_COLOR)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                dtxt = f"{dist:.1f}m" if dist else "N/A"
                cv2.putText(frame, f"{label} {conf:.2f} | {dtxt}",
                            (x1, max(y1 - 10, 15)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 1, cv2.LINE_AA)

                if dist is not None:
                    if closest_dist is None or dist < closest_dist:
                        closest_dist = dist

            # ---- TTC / AEB logic ----------------------------------------
            speed_mps = get_speed_mps(ego)
            speed_kph = speed_mps * 3.6
            ttc       = compute_ttc(closest_dist, speed_mps)

            # AEB triggers and LATCHES — stays active until vehicle is
            # fully stopped AND the driver presses 'R' to reset.
            if ttc <= TTC_BRAKE_S:
                aeb_active = True

            # ---- Keyboard → smooth racing-game controls --------------------
            key = cv2.waitKey(20) & 0xFF

            if key == ord("q") or key == 27:
                break

            # 'R' = reset AEB  (only works when vehicle is stopped)
            if key == ord("r") and aeb_active and speed_kph < 0.5:
                aeb_active = False
                throttle = 0.0
                brake    = 0.0
                logger.info("AEB reset by driver")

            if not aeb_active:
                # ---- Throttle: smooth ramp up / down ----
                if key == ord("w"):
                    throttle = min(1.0, throttle + THROTTLE_RAMP_UP)
                    brake    = max(0.0, brake - BRAKE_RAMP_DOWN * 2)
                elif key == ord("s"):
                    brake    = min(1.0, brake + BRAKE_RAMP_UP)
                    throttle = max(0.0, throttle - THROTTLE_RAMP_DOWN * 2)
                else:
                    # Release → coast: smooth decay
                    throttle = max(0.0, throttle - THROTTLE_RAMP_DOWN)
                    brake    = max(0.0, brake    - BRAKE_RAMP_DOWN)

            # ---- Steering: smooth ramp + auto-centre ----
            if key == ord("a"):
                steer = max(-1.0, steer - STEER_RAMP_UP)
            elif key == ord("d"):
                steer = min(1.0, steer + STEER_RAMP_UP)
            else:
                # Auto-centre: move steer towards 0
                if abs(steer) < STEER_CENTRE_RATE:
                    steer = 0.0
                elif steer > 0:
                    steer -= STEER_CENTRE_RATE
                else:
                    steer += STEER_CENTRE_RATE

            # Speed limiter
            if speed_kph >= MAX_SPEED_KPH:
                throttle = 0.0

            # AEB override — force brake, kill throttle, lock driving
            if aeb_active:
                throttle = 0.0
                brake    = 1.0

            ego.apply_control(carla.VehicleControl(
                throttle=throttle,
                steer=steer,
                brake=brake,
                hand_brake=aeb_active,
                reverse=False,
            ))

            # ---- FPS ----------------------------------------------------
            frame_count += 1
            now = time.time()
            if now - last_fps_time >= 1.0:
                fps = frame_count / (now - last_fps_time)
                frame_count = 0
                last_fps_time = now

            # ---- HUD overlay --------------------------------------------
            # Speed
            cv2.putText(frame, f"Speed: {speed_kph:.1f} / {MAX_SPEED_KPH:.0f} km/h",
                        (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

            # Distance to closest object
            dist_txt = f"{closest_dist:.1f}m" if closest_dist else "--"
            cv2.putText(frame, f"Closest obj: {dist_txt}",
                        (20, 80), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

            # TTC — colour-coded
            if ttc == float("inf"):
                ttc_txt   = "TTC: ---"
                ttc_color = (200, 200, 200)
            elif ttc > TTC_WARN_S:
                ttc_txt   = f"TTC: {ttc:.2f}s"
                ttc_color = (0, 255, 0)
            elif ttc > TTC_BRAKE_S:
                ttc_txt   = f"TTC: {ttc:.2f}s  WARNING"
                ttc_color = (0, 255, 255)
            else:
                ttc_txt   = f"TTC: {ttc:.2f}s  AEB BRAKING"
                ttc_color = (0, 0, 255)

            cv2.putText(frame, ttc_txt, 
                        (20, 130), cv2.FONT_HERSHEY_DUPLEX, 1.0, ttc_color, 2, cv2.LINE_AA)

            # Throttle / Brake bars
            bar_y = 160
            cv2.putText(frame, "THR:", (20, bar_y + 15),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (80, bar_y), (80 + int(200 * throttle), bar_y + 20),
                          (0, 200, 0), -1)
            cv2.rectangle(frame, (80, bar_y), (280, bar_y + 20), (100, 100, 100), 1)
            # W indicator
            w_color = (0, 255, 0) if throttle > 0.01 else (80, 80, 80)
            cv2.putText(frame, "[W]", (290, bar_y + 15),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, w_color, 1, cv2.LINE_AA)

            bar_y = 190
            cv2.putText(frame, "BRK:", (20, bar_y + 15),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (80, bar_y), (80 + int(200 * brake), bar_y + 20),
                          (0, 0, 200), -1)
            cv2.rectangle(frame, (80, bar_y), (280, bar_y + 20), (100, 100, 100), 1)
            # S indicator
            s_color = (0, 0, 255) if brake > 0.01 else (80, 80, 80)
            cv2.putText(frame, "[S]", (290, bar_y + 15),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, s_color, 1, cv2.LINE_AA)

            # AEB status bar
            if aeb_active:
                cv2.rectangle(frame, (0, CAMERA_H - 80), (CAMERA_W, CAMERA_H),
                              (0, 0, 200), -1)
                cv2.putText(frame, "AEB ACTIVE  -  BRAKE LOCKED  [R to reset]",
                            (60, CAMERA_H - 30), 
                            cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

            # Controls reminder
            cv2.putText(frame, "W: go  S: brake  A/D: steer  R: reset AEB  Q: quit",
                        (20, CAMERA_H - 100), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

            cv2.putText(frame, f"FPS: {fps:.0f}",
                        (CAMERA_W - 150, 40), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

            cv2.imshow("AEB TEST", frame)

    except KeyboardInterrupt:
        logger.info("Interrupted")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    logger.info("Cleaning up…")
    cv2.destroyAllWindows()

    for s in (rgb_cam, depth_cam):
        try:
            s.stop(); s.destroy()
        except Exception:
            pass

    if target is not None:
        try:
            target.destroy()
        except Exception:
            pass

    try:
        ego.apply_control(carla.VehicleControl(brake=1.0))
    except Exception:
        pass
    time.sleep(0.3)

    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)
    tm.set_synchronous_mode(False)

    spawner.destroy_all()
    client.disconnect()
    logger.info("Done.")


if __name__ == "__main__":
    main()

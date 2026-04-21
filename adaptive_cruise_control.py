#!/usr/bin/env python3
"""
ACC (Adaptive Cruise Control) Test — Longitudinal Only
======================================================

Test scenario:
  • Ego vehicle follows the CARLA Traffic Manager autopilot.
  • A lead vehicle is spawned ahead, also on autopilot (slower speed).
  • YOLO detects the lead vehicle; CARLA depth gives the distance.
  • The ACC controller (TTC_PID_ACC) adjusts throttle/brake to
    maintain a 15 m safe following distance.
  • Steering is handled by CARLA's autopilot — ACC only does longitudinal.

Usage:
    cd simulator_av
    python -m tests.acc.acc_test
    # or:
    cd tests/acc && python acc_test.py
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
logger = logging.getLogger("acc_test")

import cv2
import numpy as np
from ultralytics import YOLO

# Reuse perception helpers
from perception.object_detection.carla_stereo_depth import (
    carla_depth_to_meters,
    get_distance_at_bbox,
    YOLO_MODEL,
    YOLO_CONF,
    YOLO_IMGSZ,
    ALLOWED_CLASSES,
    CLASS_COLORS,
    DEFAULT_COLOR,
)

# Heading vehicle tracker (locks onto the lead in our lane)
from perception.acc.focus_heading_vehicle import HeadingVehicleTracker

# ACC controller
from control.acc.ttc_pid import TTC_PID_ACC

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

# ACC
SET_SPEED_KPH       = 30.0
SET_SPEED_MPS       = SET_SPEED_KPH / 3.6
SAFE_DISTANCE_M     = 15.0
TTC_EMERGENCY_S     = 1.5

# Lead vehicle
LEAD_DISTANCE_M     = 30.0      # initial spawn distance ahead
LEAD_SPEED_KPH      = 20.0      # slower than ego's set speed


# ============================================================================
# HELPERS
# ============================================================================

def get_speed_mps(vehicle) -> float:
    v = vehicle.get_velocity()
    return math.sqrt(v.x**2 + v.y**2 + v.z**2)


def compute_waypoint_steer(vehicle, world, look_ahead_m=5.0):
    """
    Simple lateral control: steer towards a waypoint *look_ahead_m* ahead
    on the current lane.  Returns steer in [-1, 1].
    """
    veh_tf = vehicle.get_transform()
    veh_loc = veh_tf.location
    veh_yaw = math.radians(veh_tf.rotation.yaw)

    wp = world.get_map().get_waypoint(veh_loc,
                                       project_to_road=True,
                                       lane_type=carla.LaneType.Driving)
    if wp is None:
        return 0.0

    # Get a waypoint a few metres ahead
    next_wps = wp.next(look_ahead_m)
    if not next_wps:
        return 0.0
    target = next_wps[0].transform.location

    # Angle to target
    dx = target.x - veh_loc.x
    dy = target.y - veh_loc.y
    target_yaw = math.atan2(dy, dx)

    # Heading error
    err = target_yaw - veh_yaw
    # Normalise to [-pi, pi]
    while err > math.pi:  err -= 2 * math.pi
    while err < -math.pi: err += 2 * math.pi

    # Proportional steer (gain ~2.0 works well at low speeds)
    steer = max(-1.0, min(1.0, 2.0 * err))
    return steer


def spawn_lead_vehicle(world, ego, distance_m, tm, lead_speed_kph):
    """
    Spawn a lead vehicle *distance_m* ahead of ego on the same lane.
    Driven by Traffic Manager at *lead_speed_kph*.
    """
    ego_tf = ego.get_transform()
    yaw_rad = math.radians(ego_tf.rotation.yaw)

    rough_loc = carla.Location(
        x=ego_tf.location.x + distance_m * math.cos(yaw_rad),
        y=ego_tf.location.y + distance_m * math.sin(yaw_rad),
        z=ego_tf.location.z,
    )

    # Snap to the driving lane
    waypoint = world.get_map().get_waypoint(rough_loc,
                                             project_to_road=True,
                                             lane_type=carla.LaneType.Driving)
    if waypoint is None:
        logger.error("Could not find road waypoint for lead vehicle")
        return None

    spawn_tf = waypoint.transform
    spawn_tf.location.z += 0.5

    bp_lib = world.get_blueprint_library()
    bp = bp_lib.find("vehicle.dodge.charger_police")
    if bp.has_attribute("color"):
        bp.set_attribute("color", "0,0,255")   # blue lead vehicle

    lead = world.try_spawn_actor(bp, spawn_tf)
    if lead is None:
        logger.warning("Could not spawn lead vehicle")
        return None

    # Put lead on autopilot at a slower speed
    lead.set_autopilot(True, tm.get_port())
    # Speed difference: 100% = stopped, 0% = TM speed limit.
    # We want lead_speed_kph out of the TM default (~50 km/h).
    speed_diff = 100.0 - (lead_speed_kph / 50.0 * 100.0)
    tm.vehicle_percentage_speed_difference(lead, speed_diff)
    tm.auto_lane_change(lead, False)
    tm.ignore_lights_percentage(lead, 100)
    tm.ignore_signs_percentage(lead, 100)

    logger.info(f"Lead vehicle spawned {distance_m:.0f}m ahead at "
                f"({spawn_tf.location.x:.1f}, {spawn_tf.location.y:.1f}) "
                f"— autopilot at ~{lead_speed_kph:.0f} km/h")
    return lead


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="ACC Test")
    parser.add_argument("--set-speed", type=float, default=SET_SPEED_KPH,
                        help="ACC cruise set speed (km/h)")
    parser.add_argument("--safe-dist", type=float, default=SAFE_DISTANCE_M,
                        help="Safe following distance (m)")
    parser.add_argument("--lead-speed", type=float, default=LEAD_SPEED_KPH,
                        help="Lead vehicle speed (km/h)")
    parser.add_argument("--lead-distance", type=float, default=LEAD_DISTANCE_M,
                        help="Initial lead vehicle distance (m)")
    args = parser.parse_args()

    set_speed_mps = args.set_speed / 3.6

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
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    tm = carla_client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    tm.set_global_distance_to_leading_vehicle(2.5)

    # ------------------------------------------------------------------
    # 2.  Spawn ego vehicle (autopilot for steering, ACC for throttle/brake)
    # ------------------------------------------------------------------
    spawner = VehicleSpawner(client)
    spawner._config = VehicleSpawnerConfig(default_ego_blueprint="vehicle.dodge.charger")
    ego = spawner.spawn_ego(spawn_index=1)
    if ego is None:
        logger.error("Failed to spawn ego")
        return

    # NO autopilot on ego — ACC controls throttle/brake,
    # waypoint-following controls steering.
    logger.info("Ego vehicle spawned — ACC controls throttle/brake, waypoint steering")

    # Let physics settle
    for _ in range(5):
        world.tick()
        time.sleep(0.05)

    # ------------------------------------------------------------------
    # 3.  Spawn lead vehicle
    # ------------------------------------------------------------------
    lead = spawn_lead_vehicle(world, ego, args.lead_distance, tm, args.lead_speed)

    # ------------------------------------------------------------------
    # 4.  Cameras (RGB + Depth)
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
    # 5.  YOLO + ACC controller
    # ------------------------------------------------------------------
    model = YOLO(YOLO_MODEL, task="detect")
    logger.info(f"YOLO model loaded: {YOLO_MODEL}")

    # Heading vehicle tracker — picks the lead in our lane
    tracker = HeadingVehicleTracker(
        frame_width=CAMERA_W,
        frame_height=CAMERA_H,
        lateral_band_fraction=0.4,
    )

    acc = TTC_PID_ACC(
        set_speed_mps=set_speed_mps,
        safe_distance_m=args.safe_dist,
        ttc_emergency_s=TTC_EMERGENCY_S,
        kp=0.1, ki=0.05, kd=0.001,
    )

    # ------------------------------------------------------------------
    # 6.  Warm-up
    # ------------------------------------------------------------------
    for _ in range(15):
        world.tick()
        time.sleep(0.05)
    logger.info("Warm-up done — ACC active.")

    # ------------------------------------------------------------------
    # 7.  Main loop
    # ------------------------------------------------------------------
    cv2.namedWindow("ACC TEST", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ACC TEST", CAMERA_W, CAMERA_H)
 
    fps = 0.0
    frame_count = 0
    last_fps_time = time.time()
    prev_lead_dist = None

    try:
        while True:
            world.tick()
            time.sleep(0.01)

            # ---- Grab frames -------------------------------------------
            with cb_lock:
                rgb_raw   = latest["rgb"]
                depth_raw = latest["depth"]
            if rgb_raw is None or depth_raw is None:
                continue

            frame   = np.ascontiguousarray(rgb_raw[:, :, :3])
            depth_m = carla_depth_to_meters(depth_raw)

            # ---- YOLO detection ----------------------------------------
            results = model(frame, verbose=False, conf=YOLO_CONF,
                            imgsz=YOLO_IMGSZ, device="cuda")[0]

            # Draw ALL detections (dimmed)
            for box in results.boxes:
                cls = int(box.cls[0])
                if cls not in ALLOWED_CLASSES:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf  = float(box.conf[0])
                label = model.names[cls]
                dist  = get_distance_at_bbox(depth_m, x1, y1, x2, y2)
                dtxt  = f"{dist:.1f}m" if dist else "N/A"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1, cv2.LINE_AA)
                cv2.putText(frame, f"{label} {conf:.2f} | {dtxt}",
                            (x1, max(y1 - 10, 15)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)

            # ---- Track heading vehicle (lead in our lane) ---------------
            lead_det = tracker.update(results.boxes, depth_m, model.names,
                                      get_distance_fn=get_distance_at_bbox)

            closest_dist = None
            if lead_det is not None:
                closest_dist = lead_det.distance_m
                # Highlight the lead vehicle with a bright box
                x1, y1, x2, y2 = lead_det.bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3, cv2.LINE_AA)
                dtxt = f"{lead_det.distance_m:.1f}m" if lead_det.distance_m else "N/A"
                cv2.putText(frame, f"LEAD [ID:{lead_det.track_id}] {dtxt}",
                            (x1, max(y1 - 10, 15)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            # ---- ACC update --------------------------------------------
            ego_speed = get_speed_mps(ego)

            # Estimate relative closing speed from distance change
            lead_rel_speed = None
            if closest_dist is not None and prev_lead_dist is not None:
                dt_est = 0.05
                lead_rel_speed = (prev_lead_dist - closest_dist) / dt_est
                lead_rel_speed = max(0.0, lead_rel_speed)
            prev_lead_dist = closest_dist

            acc_state = acc.update(ego_speed, closest_dist, lead_rel_speed)

            # Waypoint-based steering (keeps ego in its lane)
            steer = compute_waypoint_steer(ego, world, look_ahead_m=5.0)

            # ACC has FULL control — no autopilot fighting us
            ego.apply_control(carla.VehicleControl(
                throttle=acc_state.throttle,
                brake=acc_state.brake,
                steer=steer,
            ))

            # ---- FPS ---------------------------------------------------
            frame_count += 1
            now = time.time()
            if now - last_fps_time >= 1.0:
                fps = frame_count / (now - last_fps_time)
                frame_count = 0
                last_fps_time = now

            # ---- HUD ---------------------------------------------------
            speed_kph = ego_speed * 3.6

            # Mode color
            mode_colors = {
                "CRUISE":    (0, 255, 0),     # green
                "FOLLOW":    (0, 255, 255),   # yellow
                "EMERGENCY": (0, 0, 255),     # red
            }
            mode_color = mode_colors.get(acc_state.mode, (255, 255, 255))

            # Speed
            cv2.putText(frame, f"Speed: {speed_kph:.1f} km/h  (set: {args.set_speed:.0f})",
                        (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

            # Lead distance
            dist_txt = f"{closest_dist:.1f}m" if closest_dist else "--"
            cv2.putText(frame, f"Lead: {dist_txt}  (safe: {args.safe_dist:.0f}m)",
                        (20, 80), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

            # TTC
            if acc_state.ttc_s == float("inf"):
                ttc_txt = "TTC: ---"
            else:
                ttc_txt = f"TTC: {acc_state.ttc_s:.2f}s"
            cv2.putText(frame, ttc_txt,
                        (20, 120), cv2.FONT_HERSHEY_DUPLEX, 0.8, mode_color, 1, cv2.LINE_AA)

            # Mode banner
            cv2.putText(frame, f"ACC: {acc_state.mode}",
                        (20, 165), cv2.FONT_HERSHEY_DUPLEX, 1.0, mode_color, 2, cv2.LINE_AA)

            # Throttle / Brake bars
            bar_y = 200
            cv2.putText(frame, "THR:", (20, bar_y + 15),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (80, bar_y), (80 + int(200 * acc_state.throttle), bar_y + 20),
                          (0, 200, 0), -1)
            cv2.rectangle(frame, (80, bar_y), (280, bar_y + 20), (100, 100, 100), 1)

            bar_y = 230
            cv2.putText(frame, "BRK:", (20, bar_y + 15),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (80, bar_y), (80 + int(200 * acc_state.brake), bar_y + 20),
                          (0, 0, 200), -1)
            cv2.rectangle(frame, (80, bar_y), (280, bar_y + 20), (100, 100, 100), 1)

            # FPS
            cv2.putText(frame, f"FPS: {fps:.0f}",
                        (CAMERA_W - 150, 40),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

            # Controls
            cv2.putText(frame, "Q: quit",
                        (20, CAMERA_H - 20),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

            cv2.imshow("ACC TEST", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    except KeyboardInterrupt:
        logger.info("Interrupted")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    logger.info("Cleaning up...")
    cv2.destroyAllWindows()

    for s in (rgb_cam, depth_cam):
        try:
            s.stop(); s.destroy()
        except Exception:
            pass

    if lead is not None:
        try:
            lead.set_autopilot(False)
            lead.destroy()
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


import math
import os
import csv

import rospy
import numpy as np
import yaml

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Path
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode

import cv2
from cv_bridge import CvBridge


class FlightState:
    IDLE = "IDLE"
    TAKEOFF = "TAKEOFF"
    PATH_FOLLOW = "PATH_FOLLOW"
    LAND_APPROACH = "LAND_APPROACH"
    LAND_VISION = "LAND_VISION"
    DONE = "DONE"


class DroneDemo:
    def __init__(self):
        rospy.init_node("drone_demo", anonymous=False)

        # ----- 경로 -----
        pkg_path = os.path.expanduser("~/drone_ws/src/drone_practice")
        self.mission_file = os.path.join(pkg_path, "mission", "practice_mission.yaml")
        self.path_file = os.path.join(pkg_path, "mission", "practice_path.csv")

        # ----- 파라미터 -----
        self.takeoff_alt = 2.5
        self.lookahead = 1.0
        self.avoid_distance = 1.8
        self.avoid_gain = 1.2
        self.land_approach_alt = 1.5
        self.land_descent_speed = 0.25
        self.land_touchdown_alt = 0.3

        # ----- 미션 로드 -----
        self.waypoints = []
        self.landing_pad = None
        self.path_points = []
        self._load_mission()
        self._load_path()

        rospy.loginfo(f"Loaded {len(self.waypoints)} waypoints, "
                      f"{len(self.path_points)} path points.")

        # ----- 상태 -----
        self.state = FlightState.IDLE
        self.current_pose = None
        self.mavros_state = State()
        self.scan = None
        self.latest_image = None
        self.bridge = CvBridge()
        self.path_index = 0
        self._land_z = self.land_approach_alt

        # ----- 토픽 -----
        rospy.Subscriber("/mavros/state", State, self._state_cb)
        rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self._pose_cb)
        rospy.Subscriber("/iris/scan", LaserScan, self._scan_cb)
        rospy.Subscriber("/iris/camera_down/image_raw", Image, self._image_cb,
                         queue_size=1, buff_size=2**22)

        self.setpoint_pub = rospy.Publisher("/mavros/setpoint_position/local",
                                            PoseStamped, queue_size=10)
        self.path_viz_pub = rospy.Publisher("/demo/path_viz", Path,
                                            queue_size=1, latch=True)
        self.marker_pub = rospy.Publisher("/demo/markers", MarkerArray,
                                          queue_size=1, latch=True)
        self.detect_image_pub = rospy.Publisher("/demo/red_circle_detection", Image,
                                                queue_size=1)

        # ----- 서비스 -----
        rospy.wait_for_service("/mavros/cmd/arming")
        rospy.wait_for_service("/mavros/set_mode")
        self.arming_srv = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
        self.set_mode_srv = rospy.ServiceProxy("/mavros/set_mode", SetMode)

        self._publish_path_viz()
        self._publish_markers()

        rospy.loginfo("DroneDemo initialized.")

    # ============= 콜백 =============
    def _state_cb(self, msg): self.mavros_state = msg
    def _pose_cb(self, msg): self.current_pose = msg
    def _scan_cb(self, msg): self.scan = msg
    def _image_cb(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logwarn(f"cv_bridge error: {e}")

    # ============= 로드 =============
    def _load_mission(self):
        with open(self.mission_file, "r") as f:
            data = yaml.safe_load(f)
        m = data["mission"]
        self.takeoff_alt = float(m.get("takeoff_altitude", 2.5))
        self.waypoints = [
            (wp["position"]["x"], wp["position"]["y"], wp["position"]["z"])
            for wp in m["waypoints"]
        ]
        lp = m["landing_pad"]
        self.landing_pad = (lp["position"]["x"], lp["position"]["y"], lp["position"]["z"])

    def _load_path(self):
        with open(self.path_file, "r") as f:
            reader = csv.DictReader(f)
            self.path_points = [
                (float(r["x"]), float(r["y"]), float(r["z"]))
                for r in reader
            ]

    # ============= 시각화 =============
    def _publish_path_viz(self):
        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = rospy.Time.now()
        for x, y, z in self.path_points:
            ps = PoseStamped()
            ps.header.frame_id = "map"
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.position.z = z
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.path_viz_pub.publish(path)

    def _publish_markers(self):
        ma = MarkerArray()
        for i, (x, y, z) in enumerate(self.waypoints):
            m = Marker()
            m.header.frame_id = "map"
            m.ns = "waypoints"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.position.z = z
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 1.0
            m.color = ColorRGBA(0.2, 0.5, 1.0, 0.3)
            ma.markers.append(m)
        x, y, z = self.landing_pad
        lm = Marker()
        lm.header.frame_id = "map"
        lm.ns = "landing"
        lm.id = 0
        lm.type = Marker.CYLINDER
        lm.pose.position.x = x
        lm.pose.position.y = y
        lm.pose.position.z = 0.05
        lm.pose.orientation.w = 1.0
        lm.scale.x = 1.0
        lm.scale.y = 1.0
        lm.scale.z = 0.05
        lm.color = ColorRGBA(0.9, 0.1, 0.1, 0.8)
        ma.markers.append(lm)
        self.marker_pub.publish(ma)

    # ============= 유틸 =============
    def _dist2(self, p1, p2):
        return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

    def _publish_setpoint(self, x, y, z):
        sp = PoseStamped()
        sp.header.stamp = rospy.Time.now()
        sp.header.frame_id = "map"
        sp.pose.position.x = x
        sp.pose.position.y = y
        sp.pose.position.z = z
        sp.pose.orientation.w = 1.0
        self.setpoint_pub.publish(sp)

    def _get_xy(self):
        if self.current_pose is None:
            return None
        p = self.current_pose.pose.position
        return (p.x, p.y, p.z)

    # ============= 경로 추종 (Pure Pursuit) =============
    def _pure_pursuit_target(self):
        if not self.path_points:
            return None
        cur = self._get_xy()
        if cur is None:
            return None

        search_window = self.path_points[self.path_index:self.path_index+50]
        if not search_window:
            return self.path_points[-1]
        dists = [self._dist2(cur, p) for p in search_window]
        min_local = int(np.argmin(dists))
        self.path_index += min_local

        accum = 0.0
        idx = self.path_index
        while idx < len(self.path_points) - 1:
            d = self._dist2(self.path_points[idx], self.path_points[idx+1])
            accum += d
            if accum >= self.lookahead:
                return self.path_points[idx+1]
            idx += 1
        return self.path_points[-1]

    # ============= 회피 (APF) =============
    def _avoidance_vector(self):
        if self.scan is None or self.current_pose is None:
            return (0.0, 0.0)

        ranges = np.array(self.scan.ranges)
        angle_min = self.scan.angle_min
        angle_inc = self.scan.angle_increment

        valid = (ranges > self.scan.range_min) & (ranges < self.avoid_distance)
        if not np.any(valid):
            return (0.0, 0.0)

        idxs = np.where(valid)[0]
        rx = 0.0
        ry = 0.0
        for i in idxs:
            r = ranges[i]
            ang = angle_min + i * angle_inc
            mag = self.avoid_gain * (1.0/r - 1.0/self.avoid_distance) / (r*r)
            rx += -math.cos(ang) * mag
            ry += -math.sin(ang) * mag

        max_vec = 1.5
        n = math.hypot(rx, ry)
        if n > max_vec:
            rx = rx / n * max_vec
            ry = ry / n * max_vec
        return (rx, ry)

    # ============= 빨간 원 검출 =============
    def _detect_red_circle(self):
        if self.latest_image is None:
            return None
        img = self.latest_image.copy()
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, (0, 100, 80),  (10, 255, 255))
        mask2 = cv2.inRange(hsv, (170, 100, 80), (180, 255, 255))
        mask = cv2.bitwise_or(mask1, mask2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self._publish_detect_image(img)
            return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 200:
            self._publish_detect_image(img)
            return None
        (cx, cy), radius = cv2.minEnclosingCircle(largest)

        cv2.circle(img, (int(cx), int(cy)), int(radius), (0, 255, 0), 2)
        cv2.circle(img, (w//2, h//2), 5, (255, 0, 0), -1)
        cv2.line(img, (w//2, h//2), (int(cx), int(cy)), (0, 255, 255), 2)
        self._publish_detect_image(img)

        dx = (cx - w/2) / (w/2)
        dy = (cy - h/2) / (h/2)
        return (dx, dy)

    def _publish_detect_image(self, img):
        try:
            msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
            self.detect_image_pub.publish(msg)
        except Exception:
            pass

    # ============= 메인 루프 =============
    def run(self):
        rate = rospy.Rate(20)

        while not rospy.is_shutdown() and not self.mavros_state.connected:
            rate.sleep()
        rospy.loginfo("MAVROS connected.")

        for _ in range(50):
            self._publish_setpoint(0, 0, self.takeoff_alt)
            rate.sleep()

        self._set_offboard_and_arm()
        self.state = FlightState.TAKEOFF
        rospy.loginfo("State -> TAKEOFF")

        while not rospy.is_shutdown() and self.state != FlightState.DONE:
            self._step()
            rate.sleep()

        rospy.loginfo("Demo finished.")

    def _set_offboard_and_arm(self):
        last_req = rospy.Time.now()
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if (self.mavros_state.mode != "OFFBOARD" and
                (rospy.Time.now() - last_req).to_sec() > 2.0):
                try:
                    self.set_mode_srv(0, "OFFBOARD")
                except Exception as e:
                    rospy.logwarn(f"set_mode failed: {e}")
                last_req = rospy.Time.now()
            elif (not self.mavros_state.armed and
                  (rospy.Time.now() - last_req).to_sec() > 2.0):
                try:
                    self.arming_srv(True)
                except Exception as e:
                    rospy.logwarn(f"arming failed: {e}")
                last_req = rospy.Time.now()
            else:
                if self.mavros_state.armed and self.mavros_state.mode == "OFFBOARD":
                    rospy.loginfo("OFFBOARD + ARMED. Ready.")
                    break
            self._publish_setpoint(0, 0, self.takeoff_alt)
            rate.sleep()

    def _step(self):
        cur = self._get_xy()
        if cur is None:
            return

        if self.state == FlightState.TAKEOFF:
            self._publish_setpoint(0, 0, self.takeoff_alt)
            if abs(cur[2] - self.takeoff_alt) < 0.3:
                rospy.loginfo("Takeoff complete. -> PATH_FOLLOW")
                self.state = FlightState.PATH_FOLLOW

        elif self.state == FlightState.PATH_FOLLOW:
            target = self._pure_pursuit_target()
            if target is None:
                return
            ax, ay = self._avoidance_vector()
            self._publish_setpoint(target[0] + ax, target[1] + ay, target[2])

            if self._dist2(cur, self.landing_pad) < 1.0:
                rospy.loginfo("Near landing pad. -> LAND_APPROACH")
                self.state = FlightState.LAND_APPROACH

        elif self.state == FlightState.LAND_APPROACH:
            lx, ly, _ = self.landing_pad
            self._publish_setpoint(lx, ly, self.land_approach_alt)
            if (self._dist2(cur, (lx, ly)) < 0.3
                and abs(cur[2] - self.land_approach_alt) < 0.3):
                rospy.loginfo("Approached. -> LAND_VISION")
                self.state = FlightState.LAND_VISION
                self._land_z = self.land_approach_alt

        elif self.state == FlightState.LAND_VISION:
            lx, ly, _ = self.landing_pad
            offset = self._detect_red_circle()
            if offset is not None:
                dx_img, dy_img = offset
                world_dx = -dy_img * 0.5
                world_dy = -dx_img * 0.5
                sx = cur[0] + world_dx
                sy = cur[1] + world_dy
            else:
                sx, sy = lx, ly
            self._land_z = max(0.05, self._land_z - self.land_descent_speed / 20.0)
            self._publish_setpoint(sx, sy, self._land_z)

            if cur[2] < self.land_touchdown_alt:
                rospy.loginfo("Touchdown. -> DONE")
                try:
                    self.set_mode_srv(0, "AUTO.LAND")
                except Exception:
                    pass
                self.state = FlightState.DONE


def main():
    node = DroneDemo()
    try:
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()

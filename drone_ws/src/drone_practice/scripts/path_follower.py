#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import csv
import math
import os
import numpy as np  
import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Bool
from sensor_msgs.msg import LaserScan, Image  

class PathFollower:
    def __init__(self):
        rospy.init_node("path_follower_node")

        self.current_pose = PoseStamped()
        self.mission_started = False
        self.is_landing = False
        
        self.current_yaw = 0.0         
        
        self.max_speed = 2.0           
        self.target_alt = 2.5          
        self.arrive_dist = 1.5        
        self.waypoints = []
        
        self.stanley_k = 1.0          
        self.closest_idx = 0          
        
        self.filtered_vx = 0.0
        self.filtered_vy = 0.0
        
        self.latest_scan = None
        self.scan_angle_min = 0.0
        self.scan_angle_inc = 0.0

        # [추가] 뎁스 카메라 처리를 위한 브릿지 및 데이터 저장소
        self.bridge = CvBridge()
        self.depth_array = None
        
        # ROS Subscribers
        rospy.Subscriber("mavros/local_position/pose", PoseStamped, self.pose_cb)
        rospy.Subscriber("/mission/start_flag", Bool, self.start_cb)
        rospy.Subscriber("/laser/scan", LaserScan, self.lidar_cb)
        rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_cb) # 뎁스 카메라 구독

        # ROS Publishers
        self.vel_pub = rospy.Publisher("mavros/setpoint_velocity/cmd_vel_unstamped", Twist, queue_size=10)
        self.start_landing_pub = rospy.Publisher("/mission/start_landing", Bool, queue_size=10)

        self.rate = rospy.Rate(20)
        self.load_waypoints()

    def pose_cb(self, msg):
        self.current_pose = msg
        q = msg.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y**2 + q.z**2)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def start_cb(self, msg):
        self.mission_started = msg.data

    def lidar_cb(self, msg):
        self.latest_scan = msg.ranges
        self.scan_angle_min = msg.angle_min
        self.scan_angle_inc = msg.angle_increment

    # ==========================================
    # [추가] 뎁스 카메라 데이터를 1D 거리 배열로 변환
    # ==========================================
    def depth_cb(self, msg):
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
            cv_depth = np.nan_to_num(cv_depth, nan=10.0, posinf=10.0, neginf=0.0)
            
            h, w = cv_depth.shape
            # 피치(Pitch)로 인해 바닥을 벽으로 오인하지 않도록 화면 중앙 대역만 추출
            band_top = int(h * 0.35)
            band_bottom = int(h * 0.65)
            center_band = cv_depth[band_top:band_bottom, :]
            
            self.depth_array = np.min(center_band, axis=0)
        except Exception as e:
            pass

    # ==========================================
    # [수정] 순차적 웨이포인트 추종 (도달 시에만 인덱스 증가)
    # ==========================================
    def find_closest_idx(self, cx, cy):
        if self.closest_idx + 1 >= len(self.waypoints):
            return self.closest_idx

        target_x, target_y = self.waypoints[self.closest_idx + 1]
        dist_to_target = self.calc_distance(cx, cy, target_x, target_y)

        # 타겟 반경 1.5m 이내에 진입해야만 다음 웨이포인트로 넘어감
        if dist_to_target < 1.5:
            self.closest_idx += 1
            rospy.loginfo(f"✅ WP {self.closest_idx} 통과! 다음 경로로 이동합니다.")

        return self.closest_idx

    def get_stanley_target_angle(self, cx, cy):
        self.closest_idx = self.find_closest_idx(cx, cy)
        idx = self.closest_idx

        if idx + 1 < len(self.waypoints):
            fx, fy = self.waypoints[idx]
            nx, ny = self.waypoints[idx + 1]
            path_angle = math.atan2(ny - fy, nx - fx)
        else:
            path_angle = math.atan2(
                self.waypoints[-1][1] - self.waypoints[-2][1],
                self.waypoints[-1][0] - self.waypoints[-2][0]
            )

        wx, wy = self.waypoints[idx]
        lateral_error = ((cx - wx) * math.sin(path_angle) - (cy - wy) * math.cos(path_angle))
        
        rospy.loginfo_throttle(1.0, f"Lateral Error: {lateral_error:.2f}m, Target WP: {idx}")

        # [수정] 속도가 낮아도 제어기가 죽지 않도록 최소 속도 가중치 부여
        v_min = 0.5 
        speed = max(self.max_speed, v_min) 
        
        correction = math.atan2(self.stanley_k * lateral_error, speed)


        return path_angle + correction

    # ==========================================
    # [추가] 뎁스 카메라 기반 거대 벽 모서리 탐색 (Edge Seeking)
    # ==========================================
    def get_hybrid_target_angle(self, original_target_angle_local):
        if self.depth_array is None:
            return original_target_angle_local
            
        safe_dist = 4.0 # 4m 앞에 벽이 있는지 확인
        num_pixels = len(self.depth_array)
        if num_pixels == 0: 
            return original_target_angle_local
        
        # 화면을 좌, 중, 우 3등분
        center_region = self.depth_array[int(num_pixels*0.3) : int(num_pixels*0.7)]
        min_front = np.min(center_region) if len(center_region) > 0 else 10.0
        
        # 정면에 거대한 벽이 막고 있다면
        if min_front < safe_dist:
            left_region = self.depth_array[0 : int(num_pixels*0.3)]
            right_region = self.depth_array[int(num_pixels*0.7) : num_pixels]
            
            avg_left = np.mean(left_region) if len(left_region) > 0 else 10.0
            avg_right = np.mean(right_region) if len(right_region) > 0 else 10.0
            
            camera_fov = math.radians(80.0)
            
            # 더 넓게 뚫린 쪽(Edge)으로 타겟 각도를 강제 변경
            if avg_left > avg_right:
                rospy.loginfo_throttle(1.0, "🧱 [Depth] 거대 벽 감지: 왼쪽(Left Edge)으로 크게 우회합니다.")
                return camera_fov / 1.5 
            else:
                rospy.loginfo_throttle(1.0, "🧱 [Depth] 거대 벽 감지: 오른쪽(Right Edge)으로 크게 우회합니다.")
                return -camera_fov / 1.5
                
        # 정면이 뚫려있으면 원래의 웨이포인트 방향 유지
        return original_target_angle_local

    def compute_vfh_velocity(self, target_angle_local):
        if self.latest_scan is None:
            return self.max_speed * math.cos(target_angle_local), self.max_speed * math.sin(target_angle_local)

        num_bins = 72  
        bin_size = 2.0 * math.pi / num_bins
        histogram = [0.0] * num_bins
        
        safe_dist = 3.0       
        drone_radius = 0.6    

        for i, dist in enumerate(self.latest_scan):
            if 0.3 < dist < safe_dist and not math.isinf(dist) and not math.isnan(dist):
                angle = self.scan_angle_min + i * self.scan_angle_inc
                mag = (safe_dist - dist) ** 2  
                enlargement = math.asin(min(drone_radius / dist, 1.0))
                
                min_angle = angle - enlargement
                max_angle = angle + enlargement
                
                min_bin = int(math.floor((min_angle - (-math.pi)) / bin_size)) % num_bins
                max_bin = int(math.floor((max_angle - (-math.pi)) / bin_size)) % num_bins
                
                if min_bin <= max_bin:
                    for b in range(min_bin, max_bin + 1):
                        histogram[b] += mag
                else: 
                    for b in range(min_bin, num_bins):
                        histogram[b] += mag
                    for b in range(0, max_bin + 1):
                        histogram[b] += mag

        smoothed = [0.0] * num_bins
        for i in range(num_bins):
            val = sum(histogram[(i + j) % num_bins] for j in range(-2, 3))
            smoothed[i] = val / 5.0

        threshold = 1.0  
        open_bins = [i for i in range(num_bins) if smoothed[i] < threshold]

        if not open_bins:
            return 0.0, 0.0

        target_bin = int(math.floor((target_angle_local - (-math.pi)) / bin_size)) % num_bins
        best_bin = open_bins[0]
        min_cost = float('inf')
        
        for b in open_bins:
            diff = abs(b - target_bin)
            if diff > num_bins / 2:
                diff = num_bins - diff  
                
            if diff < min_cost:
                min_cost = diff
                best_bin = b

        chosen_angle_local = -math.pi + best_bin * bin_size + (bin_size / 2.0)
        
        deviation = min_cost * bin_size
        speed_factor = max(0.4, math.cos(deviation)) 
        
        v_x = self.max_speed * speed_factor * math.cos(chosen_angle_local)
        v_y = self.max_speed * speed_factor * math.sin(chosen_angle_local)
        
        return v_x, v_y

    def load_waypoints(self):
        import rospkg
        try:
            rospack = rospkg.RosPack()
            pkg_path = rospack.get_path('drone_practice')
            default_path = os.path.join(pkg_path, 'mission', 'practice_path.csv')
            csv_path = rospy.get_param("~csv_path", default_path)
            
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header and header[0].replace('.', '', 1).isdigit():
                    f.seek(0)
                
                for row in reader:
                    if row:
                        try:
                            x, y = float(row[0]), float(row[1])
                            self.waypoints.append((x, y))
                        except ValueError:
                            continue
            rospy.loginfo(f"Successfully loaded {len(self.waypoints)} waypoints.")
        except Exception as e:
            rospy.logerr(f"Failed to load CSV file: {e}")

    def calc_distance(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def run(self):
        while not rospy.is_shutdown():
            if not self.mission_started or len(self.waypoints) == 0:
                self.rate.sleep()
                continue

            cx = self.current_pose.pose.position.x
            cy = self.current_pose.pose.position.y
            cz = self.current_pose.pose.position.z
            
            final_x, final_y = self.waypoints[-1]
            dist_to_final = self.calc_distance(cx, cy, final_x, final_y)
            
            if dist_to_final < self.arrive_dist:
                if not self.is_landing:
                    rospy.loginfo("Path following complete! Starting Precision Landing.")
                    land_msg = Bool()
                    land_msg.data = True
                    self.start_landing_pub.publish(land_msg)
                    self.is_landing = True
                
                rospy.loginfo("Handing over control... Shutting down Path Follower.")
                rospy.signal_shutdown("Path Following Finished")
                break 

            # A. Stanley 제어기로 "이상적인 Global 방향" 계산
            target_angle_global = self.get_stanley_target_angle(cx, cy)

            # B. 기체의 현재 헤딩을 고려하여 "Local 타겟 각도"로 변환
            target_angle_local = target_angle_global - self.current_yaw
            target_angle_local = (target_angle_local + math.pi) % (2 * math.pi) - math.pi

            # ==========================================
            # [핵심 융합] 뎁스 카메라 우회 + RPLiDAR 미세 회피
            # ==========================================
            # C-1. 뎁스 카메라가 정면에 큰 벽을 감지하면 모서리 쪽으로 타겟 각도를 틀어줌
            hybrid_target_angle = self.get_hybrid_target_angle(target_angle_local)

            # C-2. RPLiDAR 기반 VFH 알고리즘이 수정된 타겟 각도를 향해 미세 장애물을 회피하며 비행 속도 계산
            raw_vx, raw_vy = self.compute_vfh_velocity(hybrid_target_angle)

            # D. 기체 안정화를 위한 속도 Low-Pass Filter
            alpha = 0.2  
            self.filtered_vx = (alpha * raw_vx) + ((1.0 - alpha) * self.filtered_vx)
            self.filtered_vy = (alpha * raw_vy) + ((1.0 - alpha) * self.filtered_vy)

            cmd_vel = Twist()
            cmd_vel.linear.x = float(np.clip(self.filtered_vx, -self.max_speed, self.max_speed))
            cmd_vel.linear.y = float(np.clip(self.filtered_vy, -self.max_speed, self.max_speed))
            rospy.loginfo_throttle(1.0, f"Command Vel -> X: {cmd_vel.linear.x:.2f}, Y: {cmd_vel.linear.y:.2f}")
            
            cmd_vel.angular.z = 0.0 
            
            alt_error = self.target_alt - cz
            cmd_vel.linear.z = float(np.clip(alt_error * 1.0, -1.0, 1.0))

            self.vel_pub.publish(cmd_vel)
            self.rate.sleep()

if __name__ == "__main__":
    try:
        pf = PathFollower()
        pf.run()
    except rospy.ROSInterruptException:
        pass

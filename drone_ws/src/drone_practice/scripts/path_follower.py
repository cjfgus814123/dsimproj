#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import csv
import math
import os
import numpy as np  
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Bool
from sensor_msgs.msg import LaserScan  

class PathFollower:
    def __init__(self):
        rospy.init_node("path_follower_node")

        self.current_pose = PoseStamped()
        self.mission_started = False
        self.is_landing = False
        
        self.current_yaw = 0.0         
        
        # 비행 및 목표 파라미터
        self.max_speed = 2.0           
        self.target_alt = 2.5          
        self.arrive_dist = 1.5        # 마지막 도착 판단 거리
        self.waypoints = []
        
        # ==========================================
        # [통합 1] Stanley 제어기 파라미터
        # ==========================================
        self.stanley_k = 1.0          # 선으로 복귀하려는 힘(Gain)
        self.closest_idx = 0          # 현재 가장 가까운 경로점 인덱스
        
        # 뒤집힘 방지를 위한 속도 필터 변수 (LPF)
        self.filtered_vx = 0.0
        self.filtered_vy = 0.0
        
        # VFH 연산을 위한 라이다 데이터 저장소
        self.latest_scan = None
        self.scan_angle_min = 0.0
        self.scan_angle_inc = 0.0
        
        # ROS Subscribers
        rospy.Subscriber("mavros/local_position/pose", PoseStamped, self.pose_cb)
        rospy.Subscriber("/mission/start_flag", Bool, self.start_cb)
        rospy.Subscriber("/laser/scan", LaserScan, self.lidar_cb)

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
    # [통합 2] Stanley Controller 로직 (목표 방향 지시)
    # ==========================================
    def find_closest_idx(self, cx, cy):
        """현재 위치에서 가장 가까운 경로점 탐색 (앞으로 전진만 하도록)"""
        min_dist = float('inf')
        best_idx = self.closest_idx

        search_end = min(len(self.waypoints), self.closest_idx + 50)
        for i in range(self.closest_idx, search_end):
            wx, wy = self.waypoints[i]
            d = self.calc_distance(cx, cy, wx, wy)
            if d < min_dist:
                min_dist = d
                best_idx = i
        return best_idx

    def get_stanley_target_angle(self, cx, cy):
        """경로 방향과 횡방향 오차를 합산하여 최적의 비행 각도(Global) 반환"""
        self.closest_idx = self.find_closest_idx(cx, cy)
        idx = self.closest_idx

        # 경로 방향(Path angle) 계산
        if idx + 1 < len(self.waypoints):
            fx, fy = self.waypoints[idx]
            nx, ny = self.waypoints[idx + 1]
            path_angle = math.atan2(ny - fy, nx - fx)
        else:
            path_angle = math.atan2(
                self.waypoints[-1][1] - self.waypoints[-2][1],
                self.waypoints[-1][0] - self.waypoints[-2][0]
            )

        # 횡방향 오차(Cross-track error) 계산
        wx, wy = self.waypoints[idx]
        lateral_error = ((cx - wx) * math.sin(path_angle) - (cy - wy) * math.cos(path_angle))

        # Stanley 보정각 합산
        speed = max(self.max_speed, 0.1)
        correction = math.atan2(self.stanley_k * lateral_error, speed)

        return path_angle + correction

    # ==========================================
    # [통합 3] Pure VFH 알고리즘 (안전한 회피 방향 및 속도 결정)
    # ==========================================
    def compute_vfh_velocity(self, target_angle_local):
        if self.latest_scan is None:
            # 라이다 데이터가 없으면 Stanley가 지시한 방향으로 직진
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
            # 완벽히 갇혔을 때: 제자리 멈춤
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
            
            # 1. 최종 목적지 도착 확인
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

            # ==========================================
            # [통합 4] 제어 파이프라인 (Stanley -> VFH -> CMD_VEL)
            # ==========================================
            # A. Stanley 제어기로 "이상적인 Global 방향" 계산
            target_angle_global = self.get_stanley_target_angle(cx, cy)

            # B. 기체의 현재 헤딩을 고려하여 "Local 타겟 각도"로 변환
            target_angle_local = target_angle_global - self.current_yaw
            target_angle_local = (target_angle_local + math.pi) % (2 * math.pi) - math.pi

            # C. VFH에 Local 타겟 각도를 넘겨주어 장애물을 피하는 "실제 비행 속도" 도출
            raw_vx, raw_vy = self.compute_vfh_velocity(target_angle_local)

            # D. 기체 안정화를 위한 속도 Low-Pass Filter
            alpha = 0.2  
            self.filtered_vx = (alpha * raw_vx) + ((1.0 - alpha) * self.filtered_vx)
            self.filtered_vy = (alpha * raw_vy) + ((1.0 - alpha) * self.filtered_vy)

            cmd_vel = Twist()
            cmd_vel.linear.x = float(np.clip(self.filtered_vx, -self.max_speed, self.max_speed))
            cmd_vel.linear.y = float(np.clip(self.filtered_vy, -self.max_speed, self.max_speed))
            
            # E. 게걸음 비행 유지 (헤딩 회전 금지)
            cmd_vel.angular.z = 0.0 
            
            # F. 고도 유지 제어
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

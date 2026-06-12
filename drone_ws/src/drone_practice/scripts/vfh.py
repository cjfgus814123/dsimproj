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
        
        # 비행 파라미터 
        self.lookahead_distance = 1.0  
        self.max_speed = 2.0           
        self.target_alt = 2.5          
        self.waypoints = []
        self.current_wp_index = 0
        
        # 드론 뒤집힘 방지를 위한 속도 필터 변수
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
        # VFH 알고리즘은 데이터를 모아서 한 번에 처리하므로 배열만 갱신합니다.
        self.latest_scan = msg.ranges
        self.scan_angle_min = msg.angle_min
        self.scan_angle_inc = msg.angle_increment

    # ==========================================
    # [핵심] VFH (Vector Field Histogram) 알고리즘 적용
    # ==========================================
    def compute_vfh_velocity(self, target_angle_local):
        if self.latest_scan is None:
            # 라이다 데이터가 없으면 목적지로 직진
            return self.max_speed * math.cos(target_angle_local), self.max_speed * math.sin(target_angle_local)

        num_bins = 72  # 360도를 5도 간격으로 나눔
        bin_size = 2.0 * math.pi / num_bins
        histogram = [0.0] * num_bins
        
        safe_dist = 3.0       # 인식 거리
        drone_radius = 0.6    # 드론의 크기 (장애물을 부풀리는 데 사용)

        # 1. 극좌표 히스토그램 생성 (Polar Histogram)
        for i, dist in enumerate(self.latest_scan):
            if 0.3 < dist < safe_dist and not math.isinf(dist) and not math.isnan(dist):
                angle = self.scan_angle_min + i * self.scan_angle_inc
                
                # 거리가 가까울수록 장애물 위험도(Magnitude) 기하급수적 증가
                mag = (safe_dist - dist) ** 2  
                
                # 드론의 크기만큼 장애물 시야각 확장 (가장 중요한 충돌 방지 로직)
                enlargement = math.asin(min(drone_radius / dist, 1.0))
                
                min_angle = angle - enlargement
                max_angle = angle + enlargement
                
                # 인덱스 매핑 (-pi ~ +pi 를 0 ~ 71 배열로 변환)
                min_bin = int(math.floor((min_angle - (-math.pi)) / bin_size)) % num_bins
                max_bin = int(math.floor((max_angle - (-math.pi)) / bin_size)) % num_bins
                
                # 장애물 구역에 위험도 덧셈
                if min_bin <= max_bin:
                    for b in range(min_bin, max_bin + 1):
                        histogram[b] += mag
                else: # 배열의 끝부분(360도 경계)을 넘어갈 때 처리
                    for b in range(min_bin, num_bins):
                        histogram[b] += mag
                    for b in range(0, max_bin + 1):
                        histogram[b] += mag

        # 2. 히스토그램 스무딩 (노이즈 제거용 이동 평균 필터)
        smoothed = [0.0] * num_bins
        for i in range(num_bins):
            val = 0
            for j in range(-2, 3):
                val += histogram[(i + j) % num_bins]
            smoothed[i] = val / 5.0

        # 3. 빈 공간(Open Valleys) 찾기
        threshold = 1.0  # 이 수치보다 낮으면 통과 가능한 빈 공간으로 간주
        open_bins = [i for i in range(num_bins) if smoothed[i] < threshold]

        if not open_bins:
            # 완벽히 갇혔을 때: 제자리 멈춤 (안전을 위해)
            return 0.0, 0.0

        # 4. 목적지 각도와 가장 가까운 빈 공간 선택
        target_bin = int(math.floor((target_angle_local - (-math.pi)) / bin_size)) % num_bins
        
        best_bin = open_bins[0]
        min_cost = float('inf')
        
        for b in open_bins:
            # 목적지 빈 공간과의 거리(Cost) 계산
            diff = abs(b - target_bin)
            if diff > num_bins / 2:
                diff = num_bins - diff  # 역방향 최단 거리 계산
                
            if diff < min_cost:
                min_cost = diff
                best_bin = b

        # 5. 선택된 빈 공간의 각도를 속도 벡터로 변환
        chosen_angle_local = -math.pi + best_bin * bin_size + (bin_size / 2.0)
        
        # 장애물을 피해 크게 돌아가야 할 경우, 속도를 살짝 줄여 안정성 확보
        deviation = min_cost * bin_size
        speed_factor = max(0.4, math.cos(deviation)) # 최소 40% 속도 보장
        
        v_x = self.max_speed * speed_factor * math.cos(chosen_angle_local)
        v_y = self.max_speed * speed_factor * math.sin(chosen_angle_local)
        
        return v_x, v_y

    def load_waypoints(self):
        csv_path = os.path.expanduser("~/catkin_ws/src/drone_ws/drone_ws/src/drone_practice/mission/practice_path.csv")
        try:
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
            
            # 1. 목적지 도착 확인
            final_x, final_y = self.waypoints[-1]
            if self.current_wp_index >= len(self.waypoints):
                if not self.is_landing:
                    rospy.loginfo("Path following complete! Starting Precision Landing.")
                    land_msg = Bool()
                    land_msg.data = True
                    self.start_landing_pub.publish(land_msg)
                    self.is_landing = True
                
                rospy.loginfo("Handing over control... Shutting down Path Follower.")
                rospy.signal_shutdown("Path Following Finished")
                break 

            # 2. 다음 웨이포인트(타겟) 선정
            target_x, target_y = self.waypoints[self.current_wp_index]
            dist_to_target = self.calc_distance(cx, cy, target_x, target_y)

            if dist_to_target < self.lookahead_distance:
                self.current_wp_index += 1
                if self.current_wp_index < len(self.waypoints):
                    target_x, target_y = self.waypoints[self.current_wp_index]

            # 3. Global 좌표계 상의 타겟 각도 계산
            dx_global = target_x - cx
            dy_global = target_y - cy
            target_angle_global = math.atan2(dy_global, dx_global)

            # 드론의 현재 방향(Yaw)을 빼서 Local 타겟 각도로 변환
            target_angle_local = target_angle_global - self.current_yaw
            target_angle_local = (target_angle_local + math.pi) % (2 * math.pi) - math.pi

            cmd_vel = Twist()
            if dist_to_target > 0:
                # ==========================================
                # VFH 알고리즘 호출 (가장 좋은 로컬 회피 속도 반환)
                # ==========================================
                raw_vx, raw_vy = self.compute_vfh_velocity(target_angle_local)

                # 뒤집힘 방지 스무딩 (Low-Pass Filter)
                alpha = 0.2  
                self.filtered_vx = (alpha * raw_vx) + ((1.0 - alpha) * self.filtered_vx)
                self.filtered_vy = (alpha * raw_vy) + ((1.0 - alpha) * self.filtered_vy)

                cmd_vel.linear.x = float(np.clip(self.filtered_vx, -self.max_speed, self.max_speed))
                cmd_vel.linear.y = float(np.clip(self.filtered_vy, -self.max_speed, self.max_speed))
                
                # ==========================================
                # 헤딩 회전 금지 (요청하신 대로 기수 고정 유지)
                # ==========================================
                cmd_vel.angular.z = 0.0 
            
            # 4. 고도 유지 제어
            alt_error = self.target_alt - cz
            cmd_vel.linear.z = alt_error * 1.0

            self.vel_pub.publish(cmd_vel)
            self.rate.sleep()

if __name__ == "__main__":
    try:
        pf = PathFollower()
        pf.run()
    except rospy.ROSInterruptException:
        pass

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
        self.max_speed = 1.5           
        self.target_alt = 2.5          
        self.waypoints = []
        self.current_wp_index = 0
        
        # [핵심 추가] 드론 뒤집힘 방지를 위한 속도 필터 변수
        self.filtered_vx = 0.0
        self.filtered_vy = 0.0
        
        # 로컬 기준 회피 속도 변수
        self.avoid_local_x = 0.0
        self.avoid_local_y = 0.0
        
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

    # ==========================================
    # [로컬 APF] 드론 기체 기준(Local)으로 장애물 밀어내기
    # ==========================================
    def lidar_cb(self, msg):
        force_x = 0.0
        force_y = 0.0
        
        safe_distance = 3.0   
        repulsive_gain = 1.0  # 뒤집힘 방지를 위해 밀어내는 힘을 약간 줄임

        for i, distance in enumerate(msg.ranges):
            # 노이즈 및 기체 자신 무시
            if distance < 0.3 or distance > msg.range_max or math.isinf(distance) or math.isnan(distance):
                continue
            
            if distance < safe_distance:
                # 라이다 각도는 이미 기체 정면 기준(Local)입니다.
                angle = msg.angle_min + i * msg.angle_increment
                
                # 가까울수록 강하게 밀어내는 힘 계산
                force = repulsive_gain * (1.0 / distance - 1.0 / safe_distance) * (1.0 / distance**2)
                force = min(force, 2.0) # 너무 튕기지 않게 제한
                
                # 드론 기준(Local)으로 장애물 반대 방향으로 힘 누적
                force_x -= force * math.cos(angle)
                force_y -= force * math.sin(angle)

        # 안전을 위해 회피 속도 제한
        self.avoid_local_x = float(np.clip(force_x, -1.0, 1.0))
        self.avoid_local_y = float(np.clip(force_y, -1.0, 1.0))
                        
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

            # 3. 타겟을 향한 Global 방향 벡터 계산
            dx_global = target_x - cx
            dy_global = target_y - cy
            dist = math.sqrt(dx_global**2 + dy_global**2)

            cmd_vel = Twist()
            if dist > 0:
                # ==========================================
                # [해결 1] Global 타겟 벡터를 드론의 Local 벡터로 회전 변환
                # ==========================================
                dx_local = dx_global * math.cos(self.current_yaw) + dy_global * math.sin(self.current_yaw)
                dy_local = -dx_global * math.sin(self.current_yaw) + dy_global * math.cos(self.current_yaw)

                target_vel_local_x = (dx_local / dist) * self.max_speed
                target_vel_local_y = (dy_local / dist) * self.max_speed

                # ==========================================
                # [해결 2] 타겟(Local) + 회피(Local)
                # ==========================================
                raw_vx = target_vel_local_x + self.avoid_local_x
                raw_vy = target_vel_local_y + self.avoid_local_y

                # ==========================================
                # [해결 3] 뒤집힘 방지 스무딩 (Low-Pass Filter)
                # 속도가 급변하지 않고 자동차 엑셀처럼 부드럽게 올라가고 내려갑니다.
                # ==========================================
                alpha = 0.2  # 0.0 ~ 1.0 사이 (작을수록 훨씬 부드럽지만 반응이 살짝 느림)
                self.filtered_vx = (alpha * raw_vx) + ((1.0 - alpha) * self.filtered_vx)
                self.filtered_vy = (alpha * raw_vy) + ((1.0 - alpha) * self.filtered_vy)

                cmd_vel.linear.x = float(np.clip(self.filtered_vx, -self.max_speed, self.max_speed))
                cmd_vel.linear.y = float(np.clip(self.filtered_vy, -self.max_speed, self.max_speed))
                
                # ---------------------------------------------------------
                # [헤딩 제어] 드론의 코가 목적지를 바라보도록 부드럽게 회전
                # ---------------------------------------------------------
                target_yaw = math.atan2(dy_global, dx_global)
                yaw_error = target_yaw - self.current_yaw
                yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi
                
                # 기수 회전 시 너무 휙 돌지 않게 속도 감소 (0.5)
                cmd_vel.angular.z = float(np.clip(yaw_error * 1.0, -0.5, 0.5))
            
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

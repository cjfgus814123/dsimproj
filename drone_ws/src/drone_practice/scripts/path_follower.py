#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import csv
import math
import os
import numpy as np  # [추가] 속도 제한(clip) 연산을 위해 추가
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Bool
from sensor_msgs.msg import LaserScan  # [추가] 라이다 데이터를 받기 위해 추가

class PathFollower:
    def __init__(self):
        rospy.init_node("path_follower_node")

        self.current_pose = PoseStamped()
        self.mission_started = False
        self.is_landing = False
        
        # Pure Pursuit 파라미터 (목적지 추종)
        self.lookahead_distance = 1.0  # 타겟을 바라보는 전방 거리 (m)
        self.max_speed = 1.5           # 드론 최고 비행 속도 (m/s)
        self.target_alt = 2.5          # 비행 고도 (m)
        self.waypoints = []
        self.current_wp_index = 0
        
        # ==========================================
        # [추가] 장애물 회피 (APF) 파라미터
        # ==========================================
        self.avoid_vel_x = 0.0
        self.avoid_vel_y = 0.0
        self.safe_distance = 3.0       # 장애물 회피를 시작할 위험 반경 (m)
        self.repulsive_gain = 1.5      # 장애물이 드론을 밀어내는 힘의 세기 (튜닝 필요)

        # ROS Subscribers
        rospy.Subscriber("mavros/local_position/pose", PoseStamped, self.pose_cb)
        rospy.Subscriber("/mission/start_flag", Bool, self.start_cb)
        
        # [추가] 라이다 센서 토픽 구독 (시뮬레이터 환경에 맞춰 토픽명 수정 필요할 수 있음)
        rospy.Subscriber("/laser/scan", LaserScan, self.lidar_cb)

        # ROS Publishers
        self.vel_pub = rospy.Publisher("mavros/setpoint_velocity/cmd_vel_unstamped", Twist, queue_size=10)
        self.start_landing_pub = rospy.Publisher("/mission/start_landing", Bool, queue_size=10)

        self.rate = rospy.Rate(20)
        self.load_waypoints()

    def pose_cb(self, msg):
        self.current_pose = msg

    def start_cb(self, msg):
        self.mission_started = msg.data

    # ==========================================
    # [추가] 라이다 콜백 함수 (장애물 회피 속도 계산)
    # ==========================================
    def lidar_cb(self, msg):
        force_x = 0.0
        force_y = 0.0

        for i, distance in enumerate(msg.ranges):
            # 너무 가깝거나(기체 노이즈) 측정 불가(무한대) 값 무시
            if distance < msg.range_min or distance > msg.range_max or math.isinf(distance) or math.isnan(distance):
                continue
            
            # 장애물이 안전 거리 안으로 들어왔을 때만 회피력 생성
            if distance < self.safe_distance:
                angle = msg.angle_min + i * msg.angle_increment
                
                # 거리가 가까워질수록 밀어내는 힘이 기하급수적으로 커지는 공식
                repulsive_force = self.repulsive_gain * (1.0 / distance - 1.0 / self.safe_distance) * (1.0 / (distance**2))
                
                # 장애물과 반대 방향(-cos, -sin)으로 힘을 누적하여 저장
                force_x -= repulsive_force * math.cos(angle)
                force_y -= repulsive_force * math.sin(angle)

        self.avoid_vel_x = force_x
        self.avoid_vel_y = force_y

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
            if len(self.waypoints) > 0:
                rospy.loginfo(f" -> 시작점: {self.waypoints[0]}")
                rospy.loginfo(f" -> 착륙점: {self.waypoints[-1]}")
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

            # 목적지 도착 확인
            if self.current_wp_index >= len(self.waypoints):
                if not self.is_landing:
                    rospy.loginfo("Path following complete! Starting Precision Landing.")
                    land_msg = Bool()
                    land_msg.data = True
                    self.start_landing_pub.publish(land_msg)
                    self.is_landing = True
                
                # 착륙 신호를 보낸 후, 제어권을 정밀 착륙 노드에 넘기기 위해 속도를 0으로 유지
                cmd_vel = Twist()
                self.vel_pub.publish(cmd_vel)
                self.rate.sleep()
                continue

            # Pure Pursuit 알고리즘
            target_x, target_y = self.waypoints[self.current_wp_index]
            dist_to_target = self.calc_distance(cx, cy, target_x, target_y)

            if dist_to_target < self.lookahead_distance:
                self.current_wp_index += 1
                if self.current_wp_index < len(self.waypoints):
                    target_x, target_y = self.waypoints[self.current_wp_index]

            dx = target_x - cx
            dy = target_y - cy
            dist = math.sqrt(dx**2 + dy**2)

            cmd_vel = Twist()
            if dist > 0:
                # 1. 목적지로 가려는 매력적인(Attractive) 속도 계산
                target_vel_x = (dx / dist) * self.max_speed
                target_vel_y = (dy / dist) * self.max_speed

                # ==========================================
                # [핵심 수정] 목적지 속도 + 장애물 회피 속도 결합
                # ==========================================
                final_vel_x = target_vel_x + self.avoid_vel_x
                final_vel_y = target_vel_y + self.avoid_vel_y

                # 안전을 위해 속도가 max_speed를 넘지 않도록 제한(Clipping)
                cmd_vel.linear.x = float(np.clip(final_vel_x, -self.max_speed, self.max_speed))
                cmd_vel.linear.y = float(np.clip(final_vel_y, -self.max_speed, self.max_speed))
            
            # 고도 유지 제어
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

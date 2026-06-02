#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import csv
import math
import os
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Bool

class PathFollower:
    def __init__(self):
        rospy.init_node("path_follower_node")

        self.current_pose = PoseStamped()
        self.mission_started = False
        self.is_landing = False
        
        # Pure Pursuit 파라미터
        self.lookahead_distance = 1.0  # 타겟을 바라보는 전방 거리 (m)
        self.max_speed = 1.5           # 드론 최고 비행 속도 (m/s)
        self.target_alt = 2.5          # 비행 고도 (m)
        self.waypoints = []
        self.current_wp_index = 0
        

        # ROS Subscribers
        rospy.Subscriber("mavros/local_position/pose", PoseStamped, self.pose_cb)
        rospy.Subscriber("/mission/start_flag", Bool, self.start_cb)

        # ROS Publishers (위치 제어가 아닌 속도 제어 사용)
        self.vel_pub = rospy.Publisher("mavros/setpoint_velocity/cmd_vel_unstamped", Twist, queue_size=10)
        self.start_landing_pub = rospy.Publisher("/mission/start_landing", Bool, queue_size=10)

        self.rate = rospy.Rate(20)
        self.load_waypoints()

    def pose_cb(self, msg):
        self.current_pose = msg

    def start_cb(self, msg):
        self.mission_started = msg.data

    def load_waypoints(self):
        # CSV 파일 경로 (실제 패키지 경로에 맞게 수정됨)
        # rospack을 사용하거나 절대경로/상대경로 활용
        csv_path = os.path.expanduser("~/catkin_ws/src/drone_ws/drone_ws/src/drone_practice/mission/practice_path.csv")
        try:
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                
                # 첫 번째 줄(헤더)이 알파벳이면 스킵하는 안전장치 추가
                header = next(reader, None)
                if header and header[0].replace('.', '', 1).isdigit():
                    # 만약 첫 줄부터 숫자라면 다시 첫 줄부터 처리하도록 되돌림
                    f.seek(0)
                
                for row in reader:
                    if row: # 빈 줄 무시
                        try:
                            x, y = float(row[0]), float(row[1])
                            self.waypoints.append((x, y))
                        except ValueError:
                            # 변환할 수 없는 글자(헤더 등)가 중간에 껴있으면 무시하고 넘어감
                            continue
            rospy.loginfo(f"Successfully loaded {len(self.waypoints)} waypoints.")
            # === [추가할 디버깅 코드] ===
            if len(self.waypoints) > 0:
                rospy.loginfo(f" -> 첫 번째 좌표 (시작점): {self.waypoints[0]}")
                rospy.loginfo(f" -> 마지막 좌표 (착륙점): {self.waypoints[-1]}")
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
            # [핵심 수정] 마지막 웨이포인트(착륙장)와의 실제 거리를 계산
            final_x, final_y = self.waypoints[-1]
            dist_to_final = self.calc_distance(cx, cy, final_x, final_y)

            # 목적지 도착 확인 (마지막 웨이포인트)
            if self.current_wp_index >= len(self.waypoints):
             if not self.is_landing:
                 rospy.loginfo("Path following complete! Starting Precision Landing.")
                 # 정밀 착륙 노드에 True 신호 발송
                 land_msg = Bool()
                 land_msg.data = True
                 self.start_landing_pub.publish(land_msg)
                 self.is_landing = True

             self.rate.sleep()
             continue

            # Pure Pursuit 알고리즘: 현재 위치에서 Lookahead 거리보다 먼 다음 타겟 찾기
            target_x, target_y = self.waypoints[self.current_wp_index]
            dist_to_target = self.calc_distance(cx, cy, target_x, target_y)

            if dist_to_target < self.lookahead_distance:
                self.current_wp_index += 1
                if self.current_wp_index < len(self.waypoints):
                    target_x, target_y = self.waypoints[self.current_wp_index]

            # 타겟을 향한 방향 벡터 계산 (Holonomic)
            dx = target_x - cx
            dy = target_y - cy
            dist = math.sqrt(dx**2 + dy**2)

            cmd_vel = Twist()
            if dist > 0:
                # 방향 벡터를 정규화(Normalize)하고 최고 속도를 곱함
                cmd_vel.linear.x = (dx / dist) * self.max_speed
                cmd_vel.linear.y = (dy / dist) * self.max_speed
            
            # 고도 유지 제어 (P 제어)
            alt_error = self.target_alt - cz
            cmd_vel.linear.z = alt_error * 1.0 # 고도 보정 계수

            self.vel_pub.publish(cmd_vel)
            self.rate.sleep()

if __name__ == "__main__":
    try:
        pf = PathFollower()
        pf.run()
    except rospy.ROSInterruptException:
        pass

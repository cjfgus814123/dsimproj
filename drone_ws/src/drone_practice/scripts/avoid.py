#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import csv
import math
import os
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool

class WaypointCommander:
    def __init__(self):
        rospy.init_node("waypoint_commander_node")

        self.current_pose = PoseStamped()
        self.waypoints = []
        self.current_wp_idx = 0
        self.target_alt = 2.5
        
        # ROS Subscribers
        rospy.Subscriber("mavros/local_position/pose", PoseStamped, self.pose_cb)

        # ROS Publishers
        # [핵심] 속도(cmd_vel) 대신, Global Planner에게 "목표 위치(Goal)"를 전달합니다!
        self.goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=10)
        self.start_landing_pub = rospy.Publisher("/mission/start_landing", Bool, queue_size=10)

        self.rate = rospy.Rate(10) # 계산이 없으니 10Hz면 충분합니다.
        self.load_waypoints()

    def pose_cb(self, msg):
        self.current_pose = msg

    def calc_distance(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def load_waypoints(self):
        # (기존 CSV 불러오는 로직 동일하게 사용)
        pass 

    def run(self):
        while not rospy.is_shutdown():
            if self.current_wp_idx >= len(self.waypoints):
                self.rate.sleep()
                continue

            cx = self.current_pose.pose.position.x
            cy = self.current_pose.pose.position.y
            cz = self.current_pose.pose.position.z  # 현재 고도 확인
            
            # ==========================================
            # [해결 1] 드론이 이륙할 때까지 대기 (고도 1.5m 이상)
            # ==========================================
            if cz < 1.5:
                rospy.loginfo_throttle(2.0, "⏳ 이륙 대기 중... (PX4 콘솔에서 'commander takeoff'를 입력해 공중으로 띄워주세요!)")
                self.rate.sleep()
                continue # 고도가 낮으면 아래의 목표 전송 로직을 무시하고 다시 처음으로 돌아감

            # ==========================================
            # (이하 기존 코드 동일)
            # ==========================================
            target_x, target_y = self.waypoints[self.current_wp_idx]
            dist_to_target = self.calc_distance(cx, cy, target_x, target_y)

            # 1. 도착 판단 로직
            if dist_to_target < 1.5:
                rospy.loginfo(f"✅ WP {self.current_wp_idx + 1} 통과!")
                self.current_wp_idx += 1
                
                if self.current_wp_idx >= len(self.waypoints):
                    rospy.loginfo("🏁 모든 경로 통과 완료! 정밀 착륙을 시작합니다.")
                    land_msg = Bool()
                    land_msg.data = True
                    self.start_landing_pub.publish(land_msg)
                    break

            # 2. Global Planner에게 목표 전송
            goal_msg = PoseStamped()
            goal_msg.header.frame_id = "map"
            goal_msg.header.stamp = rospy.Time.now()
            goal_msg.pose.position.x = target_x
            goal_msg.pose.position.y = target_y
            goal_msg.pose.position.z = self.target_alt
            
            self.goal_pub.publish(goal_msg)

            self.rate.sleep()
if __name__ == "__main__":
    try:
        wc = WaypointCommander()
        wc.run()
    except rospy.ROSInterruptException:
        pass

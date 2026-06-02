#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest
from std_msgs.msg import Bool

class MissionController:
    def __init__(self):
        rospy.init_node("mission_controller_node")
        
        self.current_state = State()
        self.current_pose = PoseStamped()
        self.takeoff_alt = 2.5  # 대회 규정 고도
        self.is_takeoff_done = False

        # ROS Subscribers
        rospy.Subscriber("mavros/state", State, self.state_cb)
        rospy.Subscriber("mavros/local_position/pose", PoseStamped, self.pose_cb)

        # ROS Publishers
        self.local_pos_pub = rospy.Publisher("mavros/setpoint_position/local", PoseStamped, queue_size=10)
        self.mission_start_pub = rospy.Publisher("/mission/start_flag", Bool, queue_size=10)

        # ROS Services
        rospy.wait_for_service("/mavros/cmd/arming")
        self.arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)
        rospy.wait_for_service("/mavros/set_mode")
        self.set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)

        self.rate = rospy.Rate(20) # 20Hz (OFFBOARD 모드 유지 필수 주기)

    def state_cb(self, msg):
        self.current_state = msg

    def pose_cb(self, msg):
        self.current_pose = msg

    def run(self):
        # 1. FCU(비행 제어기) 연결 대기
        while not rospy.is_shutdown() and not self.current_state.connected:
            self.rate.sleep()
        rospy.loginfo("Vehicle connected!")

        # 2. 이륙 목표 위치 설정 (현재 x,y 유지, z=2.5m)
        target_pose = PoseStamped()
        target_pose.pose.position.x = 0
        target_pose.pose.position.y = 0
        target_pose.pose.position.z = self.takeoff_alt

        # OFFBOARD 진입 전 Setpoint를 몇 번 보내주어야 함
        for _ in range(100):
            if rospy.is_shutdown(): break
            self.local_pos_pub.publish(target_pose)
            self.rate.sleep()

        offb_set_mode = SetModeRequest()
        offb_set_mode.custom_mode = 'OFFBOARD'
        arm_cmd = CommandBoolRequest()
        arm_cmd.value = True

        last_req = rospy.Time.now()

        # 3. 메인 루프 (이륙 -> 목표 고도 도달 시 권한 이양)
        while not rospy.is_shutdown():
            if not self.is_takeoff_done:
                # OFFBOARD 모드 전환 및 Arming
                if self.current_state.mode != "OFFBOARD" and (rospy.Time.now() - last_req) > rospy.Duration(5.0):
                    if self.set_mode_client.call(offb_set_mode).mode_sent == True:
                        rospy.loginfo("OFFBOARD enabled")
                    last_req = rospy.Time.now()
                else:
                    if not self.current_state.armed and (rospy.Time.now() - last_req) > rospy.Duration(5.0):
                        if self.arming_client.call(arm_cmd).success == True:
                            rospy.loginfo("Vehicle armed")
                        last_req = rospy.Time.now()

                # 이륙 위치로 계속 퍼블리시
                self.local_pos_pub.publish(target_pose)

                # 현재 고도가 목표 고도(2.5m)의 95% 이상 도달했는지 확인
                if self.current_pose.pose.position.z >= self.takeoff_alt * 0.95:
                    rospy.loginfo("Takeoff complete! Starting Path Following Mission.")
                    self.is_takeoff_done = True
            
            else:
                # 이륙이 완료되면, Path Follower 노드가 작동하도록 True 신호 발행
                # (이후부터는 위치 퍼블리시를 중단하고 path_follower가 속도를 퍼블리시함)
                start_msg = Bool()
                start_msg.data = True
                self.mission_start_pub.publish(start_msg)

            self.rate.sleep()

if __name__ == "__main__":
    try:
        mc = MissionController()
        mc.run()
    except rospy.ROSInterruptException:
        pass

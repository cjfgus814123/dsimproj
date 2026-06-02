#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import cv2.aruco as aruco
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, PoseStamped
from mavros_msgs.srv import SetMode, SetModeRequest
from std_msgs.msg import Bool
# [수정됨] 에러를 유발하는 Range 임포트 삭제

class PrecisionLander:
    def __init__(self):
        rospy.init_node("precision_lander_node")

        self.bridge = CvBridge()
        self.current_pose = PoseStamped()
        self.landing_started = False
        self.is_landed = False
        # ---------------------------------------------------------
        # [수정] fpv_cam.sdf 파일에 적힌 진짜 스펙으로 완벽 동기화!
        # ---------------------------------------------------------
        self.focal_length_x = 277.19 
        self.focal_length_y = 277.19 

        # 카메라 해상도 중심점 (320x240의 절반)
        self.image_center_x = 160 
        self.image_center_y = 120 
        # ---------------------------------------------------------

        # 비전 제어(P-Controller) 파라미터
        self.kp_metric = 0.05  # P (비례: 현재 오차 대응)
        self.ki_metric = 0.01 # I (적분: 바람 등 누적 오차 대응)
        self.kd_metric = 0.1  # D (미분: 급브레이크, 진동 방지)

        # 과거 데이터를 저장할 변수들
        self.error_sum_x = 0.0
        self.error_sum_y = 0.0
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
      
        self.descend_speed = -0.2 # 하강 속도 (m/s)
        self.land_alt = 0.3       # 착륙 트리거 고도

        self.debug_pub = rospy.Publisher("/camera/image_debug", Image, queue_size=1)

        # ROS Subscribers
        rospy.Subscriber("mavros/local_position/pose", PoseStamped, self.pose_cb)
        rospy.Subscriber("/iris/usb_cam/image_raw", Image, self.image_cb)
        rospy.Subscriber("/mission/start_landing", Bool, self.start_landing_cb)
        # [수정됨] 에러를 유발하는 2D 라이다(/laser/scan) 구독 삭제

        # ROS Publishers & Services
        self.vel_pub = rospy.Publisher("mavros/setpoint_velocity/cmd_vel_unstamped", Twist, queue_size=10)
        rospy.wait_for_service("/mavros/set_mode")
        self.set_mode_client = rospy.ServiceProxy("/mavros/set_mode", SetMode)

        self.rate = rospy.Rate(20)

    def pose_cb(self, msg):
        self.current_pose = msg

    def start_landing_cb(self, msg):
        self.landing_started = msg.data

    def image_cb(self, msg):
        if not self.landing_started or self.is_landed:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            height, width, _ = cv_image.shape
            self.image_center_x = width // 2
            self.image_center_y = height // 2

            # ==========================================
            # [핵심 수정] ArUco 대신 HSV 색상 공간에서 빨간색 영역 찾기
            # ==========================================
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            
            # 빨간색은 HSV 스펙트럼의 양끝(0 근처, 180 근처)에 걸쳐 있으므로 두 영역을 합쳐야 합니다.
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 100])
            upper_red2 = np.array([180, 255, 255])

            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2) # 빨간색만 흰색으로 보이는 흑백 마스크 생성

            # 흰색(빨간색) 덩어리들의 윤곽선(Contours) 찾기
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            target_found = False
            marker_center_x, marker_center_y = 0, 0

            if len(contours) > 0:
                # 화면에 빨간색이 여러 개일 수 있으니 가장 '큰' 덩어리를 찾습니다.
                largest_contour = max(contours, key=cv2.contourArea)
                
                # 아주 작은 노이즈(먼지)가 잡히는 것을 방지하기 위해 픽셀 면적이 100 이상일 때만 패드로 인정
                if cv2.contourArea(largest_contour) > 100:
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        marker_center_x = int(M["m10"] / M["m00"])
                        marker_center_y = int(M["m01"] / M["m00"])
                        target_found = True
                        
                        # (선택) 인식된 빨간 패드의 외곽선에 노란색 선 그리기
                        cv2.drawContours(cv_image, [largest_contour], -1, (0, 255, 255), 2)

            cmd_vel = Twist()
            current_z = max(0.1, self.current_pose.pose.position.z)
            
            # (이전의 'if ids is not None:' 을 'if target_found:' 로 바꿉니다)
            if target_found:
                # 1. 픽셀 오차 계산
                error_x_pixel = marker_center_x - self.image_center_x
                error_y_pixel = marker_center_y - self.image_center_y

               # 2. 핀홀 카메라 모델 적용 (현재 오차 P)
                error_x_meter = (error_x_pixel * current_z) / self.focal_length_x
                error_y_meter = (error_y_pixel * current_z) / self.focal_length_y

                # (추가) I 제어: 오차 누적 (바람 저항력)
                self.error_sum_x += error_x_meter
                self.error_sum_y += error_y_meter

                # (추가) D 제어: 오차 변화량 (브레이크 역할)
                diff_error_x = error_x_meter - self.prev_error_x
                diff_error_y = error_y_meter - self.prev_error_y

                # 4. 풀 PID 계산 후 속도 명령에 대입
                pid_x = (self.kp_metric * error_x_meter) + (self.ki_metric * self.error_sum_x) + (self.kd_metric * diff_error_x)
                pid_y = (self.kp_metric * error_y_meter) + (self.ki_metric * self.error_sum_y) + (self.kd_metric * diff_error_y)

                # 카메라 좌표계와 드론 좌표계 축 변환 적용 (x오차는 y속도로, y오차는 x속도로)
                cmd_vel.linear.x = -pid_y
                cmd_vel.linear.y = pid_x

                # 다음 루프를 위해 현재 오차를 과거 오차로 저장
                self.prev_error_x = error_x_meter
                self.prev_error_y = error_y_meter

                # 화면 중앙(오차 50픽셀 이내)에 들어오면 하강, 아니면 제자리에서 위치만 맞춤
                if abs(error_x_pixel) < 50 and abs(error_y_pixel) < 50:
                    cmd_vel.linear.z = self.descend_speed
                else:
                    cmd_vel.linear.z = 0.0

                rospy.loginfo_throttle(0.5, f"Err(X,Y): ({error_x_pixel:3d}, {error_y_pixel:3d}) | Vel(X,Y): ({cmd_vel.linear.x:5.2f}, {cmd_vel.linear.y:5.2f}) | Alt: {current_z:.2f}m")

                # (기존 코드) 중앙 십자선과 빨간 점, 오차 선 그리기
                cv2.drawMarker(cv_image, (self.image_center_x, self.image_center_y), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
                cv2.circle(cv_image, (marker_center_x, marker_center_y), 5, (0, 0, 255), -1)
                cv2.line(cv_image, (self.image_center_x, self.image_center_y), (marker_center_x, marker_center_y), (255, 0, 0), 2)
                
                # ==========================================
                # [추가] 화면 좌측 상단에 실시간 계산 데이터(HUD) 텍스트 출력
                # ==========================================
                cv2.putText(cv_image, f"1. Alt(Z)   : {current_z:.2f} m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(cv_image, f"2. Pix Err  : ({error_x_pixel}, {error_y_pixel}) px", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # 핀홀 모델로 계산된 실제 물리적 오차 (초록색)
                cv2.putText(cv_image, f"3. Meter Err: ({error_x_meter:.2f}, {error_y_meter:.2f}) m", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # P 제어로 계산된 최종 명령 속도 (빨간색)
                cv2.putText(cv_image, f"4. Cmd Vel  : ({cmd_vel.linear.x:.2f}, {cmd_vel.linear.y:.2f}) m/s", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                # ==========================================

                if current_z < self.land_alt:
                    self.trigger_auto_land()

            else:
                cv2.putText(cv_image, "RED PAD LOST - HOVERING", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cmd_vel.linear.x = 0.0
                cmd_vel.linear.y = 0.0
                cmd_vel.linear.z = 0.0

            debug_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            self.debug_pub.publish(debug_msg)
            
            self.vel_pub.publish(cmd_vel)
            
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge Error: {e}")

    def trigger_auto_land(self):
        rospy.loginfo("Low altitude reached. Triggering AUTO.LAND...")
        land_cmd = SetModeRequest()
        land_cmd.custom_mode = 'AUTO.LAND'
        if self.set_mode_client.call(land_cmd).mode_sent:
            rospy.loginfo("AUTO.LAND successful. Shutting down precision lander.")
            self.is_landed = True

    def run(self):
        rospy.loginfo("Precision Lander Node Started. Waiting for landing signal...")
        while not rospy.is_shutdown():
            self.rate.sleep()

if __name__ == "__main__":
    try:
        pl = PrecisionLander()
        pl.run()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()

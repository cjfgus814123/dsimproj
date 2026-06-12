#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import cv2.aruco as aruco
import numpy as np
import math
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, PoseStamped
from mavros_msgs.srv import SetMode, SetModeRequest
from std_msgs.msg import Bool

class PrecisionLander:
    def __init__(self):
        rospy.init_node("precision_lander_node")

        self.bridge = CvBridge()
        self.current_pose = PoseStamped()
        self.landing_started = False
        self.is_landed = False
        
        # 기체 자세 각도 (IMU)
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        
        # 카메라 렌즈 스펙
        self.focal_length_x = 277.19 
        self.focal_length_y = 277.19 
        self.image_center_x = 160 
        self.image_center_y = 120 

        # PID 게인
        self.kp_metric = 0.15  
        self.ki_metric = 0.1 
        self.kd_metric = 0.2  

        self.error_sum_x = 0.0
        self.error_sum_y = 0.0
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        
        self.descend_speed = -0.2 
        self.land_alt = 0.3       

        # ==========================================
        # [핵심] OpenCV Kalman Filter 초기화 (픽셀 필터링용)
        # 상태 벡터 X = [x, y, v_x, v_y] (4차원)
        # 측정 벡터 Z = [cx, cy] (2차원)
        # ==========================================
        self.kf = cv2.KalmanFilter(4, 2)
        
        # 관측 행렬 (H): 측정값 Z가 상태 X의 어떤 부분인지 정의 [x, y 부분만 추출]
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)
        
        # 상태 전이 행렬 (A): 예측 모델 (등속도 모델, dt는 매 프레임 업데이트됨)
        # 초기에는 dt = 0.033 (약 30Hz)으로 세팅
        self.kf.transitionMatrix = np.array([
            [1, 0, 0.033, 0],
            [0, 1, 0, 0.033],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)
        
        # 예측 노이즈 (Q): 물리 모델 불확실성 (값이 클수록 예측보다 센서를 믿음)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        
        # 측정 노이즈 (R): 카메라/YOLO 흔들림 (값이 클수록 센서를 덜 믿고 부드러워짐)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        
        # 오차 공분산 초기화 (P)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0
        
        self.last_time = rospy.Time.now()
        self.kf_initialized = False # 칼만 필터가 첫 프레임을 받았는지 확인

        self.debug_pub = rospy.Publisher("/camera/image_debug", Image, queue_size=1)

        rospy.Subscriber("mavros/local_position/pose", PoseStamped, self.pose_cb)
        rospy.Subscriber("/iris/usb_cam/image_raw", Image, self.image_cb)
        rospy.Subscriber("/mission/start_landing", Bool, self.start_landing_cb)

        self.vel_pub = rospy.Publisher("mavros/setpoint_velocity/cmd_vel_unstamped", Twist, queue_size=10)
        rospy.wait_for_service("/mavros/set_mode")
        self.set_mode_client = rospy.ServiceProxy("/mavros/set_mode", SetMode)

        self.rate = rospy.Rate(20)

    def pose_cb(self, msg):
        self.current_pose = msg
        q = msg.pose.orientation
        
        sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1 - 2 * (q.x**2 + q.y**2)
        self.roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (q.w * q.y - q.z * q.x)
        if abs(sinp) >= 1:
            self.pitch = math.copysign(math.pi / 2, sinp)
        else:
            self.pitch = math.asin(sinp)

        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y**2 + q.z**2)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

    def start_landing_cb(self, msg):
        self.landing_started = msg.data

    def image_cb(self, msg):
        if not self.landing_started or self.is_landed:
            self.last_time = rospy.Time.now()
            return

        try:
            current_time = rospy.Time.now()
            dt = (current_time - self.last_time).to_sec()
            if dt <= 0: dt = 0.033
            self.last_time = current_time

            # 칼만 필터의 A(상태전이행렬)에 현재 dt 반영
            self.kf.transitionMatrix[0, 2] = dt
            self.kf.transitionMatrix[1, 3] = dt

            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            height, width, _ = cv_image.shape
            self.image_center_x = width // 2
            self.image_center_y = height // 2

            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 100])
            upper_red2 = np.array([180, 255, 255])

            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            target_found = False
            raw_cx, raw_cy = 0, 0

            if len(contours) > 0:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > 100:
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        raw_cx = int(M["m10"] / M["m00"])
                        raw_cy = int(M["m01"] / M["m00"])
                        target_found = True
                        cv2.drawContours(cv_image, [largest_contour], -1, (0, 255, 255), 2)

            cmd_vel = Twist()
            current_z = max(0.1, self.current_pose.pose.position.z)
            
            # ==========================================
            # OpenCV Kalman Filter: 예측 및 업데이트 (픽셀 기준)
            # ==========================================
            # 항상 미래 위치를 먼저 예측 (Prediction)
            predicted = self.kf.predict()
            kf_cx, kf_cy = predicted[0][0], predicted[1][0]

            if target_found:
                # 첫 발견 시, 필터가 엉뚱한 곳에서 시작하지 않도록 위치 강제 초기화
                if not self.kf_initialized:
                    self.kf.statePost = np.array([[raw_cx], [raw_cy], [0], [0]], np.float32)
                    kf_cx, kf_cy = raw_cx, raw_cy
                    self.kf_initialized = True
                else:
                    # 측정이 들어오면 Z 행렬을 만들어 Update
                    measurement = np.array([[np.float32(raw_cx)], [np.float32(raw_cy)]])
                    estimated = self.kf.correct(measurement)
                    kf_cx, kf_cy = estimated[0][0], estimated[1][0]
            else:
                # 타겟을 놓친 경우, update 없이 predict된 값(관성)을 그대로 믿음
                pass

            # ==========================================
            # 보정된 픽셀 좌표(kf_cx, kf_cy)로 물리 오차 제어
            # ==========================================
            if self.kf_initialized:
                error_x_pixel = kf_cx - self.image_center_x
                error_y_pixel = kf_cy - self.image_center_y

                # 핀홀 모델 (보정된 픽셀 -> 물리적 미터)
                raw_x_meter = (error_x_pixel * current_z) / self.focal_length_x
                raw_y_meter = (error_y_pixel * current_z) / self.focal_length_y

                # IMU 자세 보상 (가짜 오차 제거)
                comp_x = current_z * math.tan(self.pitch)
                comp_y = current_z * math.tan(self.roll)

                true_x_meter = raw_x_meter - comp_x
                true_y_meter = raw_y_meter - comp_y

                # PID 제어부
                self.error_sum_x = np.clip(self.error_sum_x + true_x_meter, -1.0, 1.0)
                self.error_sum_y = np.clip(self.error_sum_y + true_y_meter, -1.0, 1.0)

                if abs(error_x_pixel) < 10 and abs(error_y_pixel) < 10:
                    self.error_sum_x = 0.0
                    self.error_sum_y = 0.0

                diff_error_x = true_x_meter - self.prev_error_x
                diff_error_y = true_y_meter - self.prev_error_y

                pid_x = (self.kp_metric * true_x_meter) + (self.ki_metric * self.error_sum_x) + (self.kd_metric * diff_error_x)
                pid_y = (self.kp_metric * true_y_meter) + (self.ki_metric * self.error_sum_y) + (self.kd_metric * diff_error_y)

                # 90도 회전 카메라 매핑
                cmd_vel.linear.x = float(np.clip(pid_x, -1.0, 1.0))
                cmd_vel.linear.y = float(np.clip(-pid_y, -1.0, 1.0))

                self.prev_error_x = true_x_meter
                self.prev_error_y = true_y_meter

                # 하강 판정
                if abs(error_x_pixel) < 80 and abs(error_y_pixel) < 80:
                    cmd_vel.linear.z = self.descend_speed
                else:
                    cmd_vel.linear.z = -0.05 
                    
                rospy.loginfo_throttle(0.5, f"Raw Pix:({raw_cx},{raw_cy}) | KF Pix:({int(kf_cx)},{int(kf_cy)}) | Vel:({cmd_vel.linear.x:.2f},{cmd_vel.linear.y:.2f})")

            else:
                cmd_vel.linear.x = 0.0
                cmd_vel.linear.y = 0.0
                cmd_vel.linear.z = -0.05 
                self.error_sum_x, self.error_sum_y = 0.0, 0.0
                self.prev_error_x, self.error_y_meter = 0.0, 0.0

            # HUD 드로잉: 원본 YOLO(빨강) vs KF 추정치(초록) 비교
            cv2.drawMarker(cv_image, (self.image_center_x, self.image_center_y), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
            if target_found:
                cv2.circle(cv_image, (raw_cx, raw_cy), 5, (0, 0, 255), -1) # Raw: 빨간 점
            if self.kf_initialized:
                cv2.circle(cv_image, (int(kf_cx), int(kf_cy)), 8, (0, 255, 0), 2) # KF: 초록색 큰 원
                cv2.line(cv_image, (self.image_center_x, self.image_center_y), (int(kf_cx), int(kf_cy)), (255, 0, 0), 2)

            cv2.putText(cv_image, f"1. Alt(Z)   : {current_z:.2f} m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(cv_image, f"2. KF Pixel : ({int(kf_cx)}, {int(kf_cy)})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(cv_image, f"3. Cmd Vel  : ({cmd_vel.linear.x:.2f}, {cmd_vel.linear.y:.2f}) m/s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if not target_found and self.kf_initialized:
                cv2.putText(cv_image, "PAD LOST - KF TRACKING...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

            if current_z < self.land_alt:
                # 완벽하게 중심(오차 15픽셀 이내)에 들어왔는지 확인
                if abs(error_x_pixel) < 15 and abs(error_y_pixel) < 15:
                    self.trigger_auto_land()
                else:
                    # 아직 중심에 못 왔다면, 더 이상 하강하지 말고 그 고도에서 호버링하며 중심을 맞춤!
                    cmd_vel.linear.z = 0.0 
                    cv2.putText(cv_image, "ALIGNING FOR LAND...", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

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

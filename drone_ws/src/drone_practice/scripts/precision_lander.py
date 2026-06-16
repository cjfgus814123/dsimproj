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
        # [핵심] OpenCV Kalman Filter 초기화
        # ==========================================
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)
        
        self.kf.transitionMatrix = np.array([
            [1, 0, 0.033, 0],
            [0, 1, 0, 0.033],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)
        
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0
        
        self.last_time = rospy.Time.now()
        self.kf_initialized = False 

        self.debug_pub = rospy.Publisher("/camera/image_debug", Image, queue_size=1)

        rospy.Subscriber("mavros/local_position/pose", PoseStamped, self.pose_cb)
        rospy.Subscriber("/iris/usb_cam/image_raw", Image, self.image_cb)
        rospy.Subscriber("/mission/start_landing", Bool, self.start_landing_cb)

        self.vel_pub = rospy.Publisher("mavros/setpoint_velocity/cmd_vel_unstamped", Twist, queue_size=10)
        
        # ==========================================
        # [추가] path_follower 에게 패드 발견 여부를 알리는 Publisher
        # ==========================================
        self.pad_detected_pub = rospy.Publisher("/vision/pad_detected", Bool, queue_size=1)

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
        # 이미 착륙 완료 상태면 CPU 절약을 위해 종료
        if self.is_landed:
            return

        try:
            # ----------------------------------------------------
            # 1. 감시 모드 (착륙 명령 전에도 항상 바닥을 분석하여 패드를 찾습니다)
            # ----------------------------------------------------
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

            # [핵심] 패드를 찾았는지 여부를 path_follower에게 매 프레임 알려줍니다!
            detect_msg = Bool()
            detect_msg.data = target_found
            self.pad_detected_pub.publish(detect_msg)

            # [추가] 착륙 명령을 받기 전이라면 제어 명령은 쏘지 않고 화상 디버깅만 퍼블리시
            if not self.landing_started:
                self.last_time = rospy.Time.now()
                if target_found:
                    cv2.putText(cv_image, "PAD SPOTTED! WAITING HANDOVER...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(cv_image, "SEARCHING PAD...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                
                debug_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
                self.debug_pub.publish(debug_msg)
                return

            # ----------------------------------------------------
            # 2. 제어 모드 (착륙 명령이 떨어진 이후 실행되는 기존 로직)
            # ----------------------------------------------------
            current_time = rospy.Time.now()
            dt = (current_time - self.last_time).to_sec()
            if dt <= 0: dt = 0.033
            self.last_time = current_time

            self.kf.transitionMatrix[0, 2] = dt
            self.kf.transitionMatrix[1, 3] = dt

            cmd_vel = Twist()
            current_z = max(0.1, self.current_pose.pose.position.z)
            
            predicted = self.kf.predict()
            kf_cx, kf_cy = predicted[0][0], predicted[1][0]

            if target_found:
                if not self.kf_initialized:
                    self.kf.statePost = np.array([[raw_cx], [raw_cy], [0], [0]], np.float32)
                    kf_cx, kf_cy = raw_cx, raw_cy
                    self.kf_initialized = True
                else:
                    measurement = np.array([[np.float32(raw_cx)], [np.float32(raw_cy)]])
                    estimated = self.kf.correct(measurement)
                    kf_cx, kf_cy = estimated[0][0], estimated[1][0]
            else:
                pass

            if self.kf_initialized:
                error_x_pixel = kf_cx - self.image_center_x
                error_y_pixel = kf_cy - self.image_center_y

                raw_x_meter = (error_x_pixel * current_z) / self.focal_length_x
                raw_y_meter = (error_y_pixel * current_z) / self.focal_length_y

                comp_x = current_z * math.tan(self.pitch)
                comp_y = current_z * math.tan(self.roll)

                true_x_meter = raw_x_meter - comp_x
                true_y_meter = raw_y_meter - comp_y

                self.error_sum_x = np.clip(self.error_sum_x + true_x_meter, -1.0, 1.0)
                self.error_sum_y = np.clip(self.error_sum_y + true_y_meter, -1.0, 1.0)

                if abs(error_x_pixel) < 10 and abs(error_y_pixel) < 10:
                    self.error_sum_x = 0.0
                    self.error_sum_y = 0.0

                diff_error_x = true_x_meter - self.prev_error_x
                diff_error_y = true_y_meter - self.prev_error_y

                pid_x = (self.kp_metric * true_x_meter) + (self.ki_metric * self.error_sum_x) + (self.kd_metric * diff_error_x)
                pid_y = (self.kp_metric * true_y_meter) + (self.ki_metric * self.error_sum_y) + (self.kd_metric * diff_error_y)

                cmd_vel.linear.x = float(np.clip(pid_x, -1.0, 1.0))
                cmd_vel.linear.y = float(np.clip(-pid_y, -1.0, 1.0))

                self.prev_error_x = true_x_meter
                self.prev_error_y = true_y_meter

                if abs(error_x_pixel) < 80 and abs(error_y_pixel) < 80:
                    cmd_vel.linear.z = self.descend_speed
                else:
                    cmd_vel.linear.z = -0.05 
                    
                rospy.loginfo_throttle(0.5, f"Raw Pix:({raw_cx},{raw_cy}) | KF Pix:({int(kf_cx)},{int(kf_cy)}) | Vel:({cmd_vel.linear.x:.2f},{cmd_vel.linear.y:.2f})")

            else:
                cmd_vel.linear.x = 0.0
                cmd_vel.linear.y = 0.0
                cmd_vel.linear.z = -0.02 
                self.error_sum_x, self.error_sum_y = 0.0, 0.0
                self.prev_error_x, self.error_y_meter = 0.0, 0.0

            cv2.drawMarker(cv_image, (self.image_center_x, self.image_center_y), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
            if target_found:
                cv2.circle(cv_image, (raw_cx, raw_cy), 5, (0, 0, 255), -1) 
            if self.kf_initialized:
                cv2.circle(cv_image, (int(kf_cx), int(kf_cy)), 8, (0, 255, 0), 2) 
                cv2.line(cv_image, (self.image_center_x, self.image_center_y), (int(kf_cx), int(kf_cy)), (255, 0, 0), 2)

            cv2.putText(cv_image, f"1. Alt(Z)   : {current_z:.2f} m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(cv_image, f"2. KF Pixel : ({int(kf_cx)}, {int(kf_cy)})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(cv_image, f"3. Cmd Vel  : ({cmd_vel.linear.x:.2f}, {cmd_vel.linear.y:.2f}) m/s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if not target_found and self.kf_initialized:
                cv2.putText(cv_image, "PAD LOST - KF TRACKING...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

            if current_z < self.land_alt:
                if abs(error_x_pixel) < 15 and abs(error_y_pixel) < 15:
                    self.trigger_auto_land()
                else:
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

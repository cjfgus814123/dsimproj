#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import csv
import math
import os
import numpy as np  
import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist, PoseStamped, Point
from std_msgs.msg import Bool
from sensor_msgs.msg import LaserScan, Image  
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker

class PathFollower:
    def __init__(self):
        rospy.init_node("path_follower_node")

        self.current_pose = PoseStamped()
        self.mission_started = False
        self.is_landing = False
        
        self.wall_avoid_direction = 0  
        self.wall_clear_count = 0      
        
        self.current_yaw = 0.0         
        
        self.max_speed = 1.8           
        self.target_alt = 2.5          
        self.arrive_dist = 1.5        
        self.waypoints = []
        
        self.stanley_k = 1.5          
        self.closest_idx = 0          
        
        self.filtered_vx = 0.0
        self.filtered_vy = 0.0
        
        self.latest_scan = None
        self.scan_angle_min = 0.0
        self.scan_angle_inc = 0.0

        self.bridge = CvBridge()
        self.depth_array = None
        
        self.pad_detected = False
        
        self.global_path_pub = rospy.Publisher('/rviz/global_path', Path, queue_size=1, latch=True)
        self.drone_traj_pub = rospy.Publisher('/rviz/drone_trajectory', Path, queue_size=10)
        self.vfh_marker_pub = rospy.Publisher('/rviz/vfh_vector', Marker, queue_size=10)
        
        self.drone_path_msg = Path()
        self.drone_path_msg.header.frame_id = "odom"

        rospy.Subscriber("mavros/local_position/pose", PoseStamped, self.pose_cb)
        rospy.Subscriber("/mission/start_flag", Bool, self.start_cb)
        rospy.Subscriber("/laser/scan", LaserScan, self.lidar_cb)
        rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_cb)
        rospy.Subscriber("/vision/pad_detected", Bool, self.pad_detect_cb)

        self.vel_pub = rospy.Publisher("mavros/setpoint_velocity/cmd_vel_unstamped", Twist, queue_size=10)
        self.start_landing_pub = rospy.Publisher("/mission/start_landing", Bool, queue_size=10)

        self.rate = rospy.Rate(15)
        self.load_waypoints()
        
        self.publish_global_path()

    def pose_cb(self, msg):
        self.current_pose = msg
        q = msg.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y**2 + q.z**2)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)
        
        self.drone_path_msg.poses.append(msg)
        self.drone_traj_pub.publish(self.drone_path_msg)

    def start_cb(self, msg):
        self.mission_started = msg.data

    def pad_detect_cb(self, msg):
        self.pad_detected = msg.data

    def lidar_cb(self, msg):
        self.latest_scan = msg.ranges
        self.scan_angle_min = msg.angle_min
        self.scan_angle_inc = msg.angle_increment

    def depth_cb(self, msg):
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
            cv_depth = np.nan_to_num(cv_depth, nan=10.0, posinf=10.0, neginf=0.0)
            
            h, w = cv_depth.shape
            band_top = int(h * 0.35)
            band_bottom = int(h * 0.65)
            center_band = cv_depth[band_top:band_bottom, :]
            
            self.depth_array = np.min(center_band, axis=0)
        except Exception:
            pass

    def publish_global_path(self):
        path_msg = Path()
        path_msg.header.frame_id = "odom"
        path_msg.header.stamp = rospy.Time.now()
        for wp in self.waypoints:
            pose = PoseStamped()
            pose.pose.position.x = wp[0]
            pose.pose.position.y = wp[1]
            pose.pose.position.z = self.target_alt
            path_msg.poses.append(pose)
        self.global_path_pub.publish(path_msg)

    def publish_vfh_marker(self, cx, cy, cz, vx, vy):
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "vfh_vector"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        tail = Point()
        tail.x, tail.y, tail.z = cx, cy, cz
        head = Point()
        head.x = cx + vx * 1.5 
        head.y = cy + vy * 1.5
        head.z = cz
        
        marker.points = [tail, head]
        marker.scale.x = 0.1 
        marker.scale.y = 0.2 
        marker.color.a = 1.0
        marker.color.r, marker.color.g, marker.color.b = 0.0, 1.0, 0.0 
        
        self.vfh_marker_pub.publish(marker)

    def find_closest_idx(self, cx, cy):
        if self.closest_idx + 1 >= len(self.waypoints):
            return self.closest_idx

        wp_current = self.waypoints[self.closest_idx]
        wp_next = self.waypoints[self.closest_idx + 1]

        strict_arrive_dist = 0.5 
        dist_to_next = self.calc_distance(cx, cy, wp_next[0], wp_next[1])
        if dist_to_next < strict_arrive_dist:
            self.closest_idx += 1
            rospy.loginfo(f"✅ WP {self.closest_idx} 반경 내 도착! 다음 경로로 이동합니다.")
            return self.closest_idx

        vec_path_x = wp_next[0] - wp_current[0]
        vec_path_y = wp_next[1] - wp_current[1]
        vec_drone_x = cx - wp_current[0]
        vec_drone_y = cy - wp_current[1]

        path_length_sq = vec_path_x**2 + vec_path_y**2
        path_length = math.sqrt(path_length_sq)
        
        if path_length > 0:
            dot_product = vec_path_x * vec_drone_x + vec_path_y * vec_drone_y
            cross_product = abs(vec_path_x * vec_drone_y - vec_path_y * vec_drone_x)
            lateral_error = cross_product / path_length
            
            gate_width = 1.2 
            
            if dot_product > path_length_sq and lateral_error < gate_width:
                self.closest_idx += 1
                rospy.loginfo(f"⏩ WP {self.closest_idx} 게이트 통과 (오차: {lateral_error:.2f}m)! 다음 점으로 전환합니다.")

        return self.closest_idx

    def get_stanley_target_angle(self, cx, cy):
        self.closest_idx = self.find_closest_idx(cx, cy)
        idx = self.closest_idx

        if idx + 1 < len(self.waypoints):
            fx, fy = self.waypoints[idx]
            nx, ny = self.waypoints[idx + 1]
            path_angle = math.atan2(ny - fy, nx - fx)
        else:
            path_angle = math.atan2(
                self.waypoints[-1][1] - self.waypoints[-2][1],
                self.waypoints[-1][0] - self.waypoints[-2][0]
            )

        wx, wy = self.waypoints[idx]
        lateral_error = ((cx - wx) * math.sin(path_angle) - (cy - wy) * math.cos(path_angle))
        
        v_min = 0.5 
        speed = max(self.max_speed, v_min) 
        correction = math.atan2(self.stanley_k * lateral_error, speed)

        return path_angle + correction

    # ==========================================
    # [핵심 수정] 완벽한 90도 직각 게걸음 및 넉넉한 락(Lock) 해제
    # ==========================================
    def get_hybrid_target_angle(self, original_target_angle_local):
        if self.depth_array is None:
            return original_target_angle_local
            
        safe_dist = 4.0 
        num_pixels = len(self.depth_array)
        if num_pixels == 0: 
            return original_target_angle_local
        
        center_region = self.depth_array[int(num_pixels*0.3) : int(num_pixels*0.7)]
        min_front = np.min(center_region) if len(center_region) > 0 else 10.0
        
        # [수정] 대각선이 아니라 완벽한 90도(직각)로 설정하여 옆으로만 밀고 나가게 만듦!
        detour_angle = math.pi / 2.0 
        
        if min_front < safe_dist:
            self.wall_clear_count = 0  
            
            if self.wall_avoid_direction == 0:
                left_region = self.depth_array[0 : int(num_pixels*0.3)]
                right_region = self.depth_array[int(num_pixels*0.7) : num_pixels]
                
                avg_left = np.mean(left_region) if len(left_region) > 0 else 10.0
                avg_right = np.mean(right_region) if len(right_region) > 0 else 10.0
                
                if avg_left > avg_right:
                    self.wall_avoid_direction = 1
                    rospy.loginfo("🧱 [Depth] 거대 벽 감지: [왼쪽]으로 완벽한 게걸음 락(Lock)!")
                else:
                    self.wall_avoid_direction = -1
                    rospy.loginfo("🧱 [Depth] 거대 벽 감지: [오른쪽]으로 완벽한 게걸음 락(Lock)!")
                    
            if self.wall_avoid_direction == 1:
                return detour_angle
            else:
                return -detour_angle
                
        else:
            if self.wall_avoid_direction != 0:
                self.wall_clear_count += 1
                
                # [수정] 20프레임(1초) 동안 뚫린 것을 확인한 후에야 락을 풀어 확실하게 벽을 벗어남
                if self.wall_clear_count > 30:
                    self.wall_avoid_direction = 0
                    rospy.loginfo("✅ [Depth] 거대 벽 완벽 탈출 완료. 정상 경로로 복귀합니다.")
                else:
                    return detour_angle if self.wall_avoid_direction == 1 else -detour_angle
                    
        return original_target_angle_local

    def compute_vfh_velocity(self, target_angle_local):
        if self.latest_scan is None:
            return self.max_speed * math.cos(target_angle_local), self.max_speed * math.sin(target_angle_local)
        # 1. 벽 추종용 로직 추가
        # 라이다 데이터 중 왼쪽/오른쪽 벽과의 거리 확인
        left_side = np.array(self.latest_scan[0:20]) # 라이다의 왼쪽 90도 범위
        right_side = np.array(self.latest_scan[52:72]) # 라이다의 오른쪽 90도 범위
        
        # [핵심] 벽과의 목표 거리 (0.6m)
        target_wall_dist = 0.6 
        
        # 벽을 타고 가야 하는 상황인지 판단 (전방 1.5m 이내에 벽이 있는 경우)
        front_dist = np.min(self.latest_scan[30:42])
        
        # 벽 추종 제어 명령
        wall_vx, wall_vy = 0.0, 0.0
        
        if front_dist < 1.5:
            # 왼쪽 벽이 더 가깝다면 오른쪽으로 살짝 밀어주며 전진
            if np.min(left_side) < np.min(right_side):
                wall_vy = -0.5 # 오른쪽으로 게걸음
                wall_vx = 0.3  # 느리게 전진
            else:
                wall_vy = 0.5  # 왼쪽으로 게걸음
                wall_vx = 0.3
            return wall_vx, wall_vy
            
        num_bins = 72  
        bin_size = 2.0 * math.pi / num_bins
        histogram = [0.0] * num_bins
        
        # [수정] 옆으로 미끄러질 때 VFH가 과민반응하지 않도록 안전 반경을 살짝 축소
        safe_dist = 2.0        
        drone_radius = 0.4    

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
            if self.is_landing:
                self.rate.sleep()
                continue

            if not self.mission_started or len(self.waypoints) == 0:
                self.rate.sleep()
                continue

            cx = self.current_pose.pose.position.x
            cy = self.current_pose.pose.position.y
            cz = self.current_pose.pose.position.z
            
            final_x, final_y = self.waypoints[-1]
            dist_to_final = self.calc_distance(cx, cy, final_x, final_y)
            is_final_leg = (self.closest_idx + 1 >= len(self.waypoints))
            
            current_max_speed = self.max_speed
            if is_final_leg and dist_to_final < 5.0:
                current_max_speed = max(0.5, self.max_speed * (dist_to_final / 5.0))
            
            if is_final_leg and self.pad_detected:
                rospy.loginfo("🎯 [Vision] 랜딩 패드 포착! 제어권을 조기 인계합니다.")
                land_msg = Bool()
                land_msg.data = True
                self.start_landing_pub.publish(land_msg)
                self.is_landing = True
                rospy.sleep(1.0)
                continue

            if is_final_leg and dist_to_final < self.arrive_dist:
                rospy.loginfo("Path following complete! Starting Precision Landing.")
                land_msg = Bool()
                land_msg.data = True
                self.start_landing_pub.publish(land_msg)
                self.is_landing = True
                rospy.sleep(1.0)
                continue

            lookahead_time = 0.3  
            pred_cx = cx + (self.filtered_vx * lookahead_time)
            pred_cy = cy + (self.filtered_vy * lookahead_time)

            target_angle_global = self.get_stanley_target_angle(pred_cx, pred_cy)
            target_angle_local = target_angle_global - self.current_yaw
            target_angle_local = (target_angle_local + math.pi) % (2 * math.pi) - math.pi

            hybrid_target_angle = self.get_hybrid_target_angle(target_angle_local)
            raw_vx, raw_vy = self.compute_vfh_velocity(hybrid_target_angle)

            alpha = 0.5  
            self.filtered_vx = (alpha * raw_vx) + ((1.0 - alpha) * self.filtered_vx)
            self.filtered_vy = (alpha * raw_vy) + ((1.0 - alpha) * self.filtered_vy)

            cmd_vel = Twist()
            cmd_vel.linear.x = float(np.clip(self.filtered_vx, -current_max_speed, current_max_speed))
            cmd_vel.linear.y = float(np.clip(self.filtered_vy, -current_max_speed, current_max_speed))
            
            # [유지] 게걸음 비행 (기수 회전 금지)
            cmd_vel.angular.z = 0.0 
            
            alt_error = self.target_alt - cz
            cmd_vel.linear.z = float(np.clip(alt_error * 1.0, -1.0, 1.0))

            self.vel_pub.publish(cmd_vel)

            global_vx = cmd_vel.linear.x * math.cos(self.current_yaw) - cmd_vel.linear.y * math.sin(self.current_yaw)
            global_vy = cmd_vel.linear.x * math.sin(self.current_yaw) + cmd_vel.linear.y * math.cos(self.current_yaw)
            
            self.publish_vfh_marker(cx, cy, cz, global_vx, global_vy)
            
            self.rate.sleep()

if __name__ == "__main__":
    try:
        pf = PathFollower()
        pf.run()
    except rospy.ROSInterruptException:
        pass

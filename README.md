Autonomous UAV Navigation & Precision Landing System
---
프로젝트 개요 (Project Overview)
---
이 프로젝트는 PX4 Autopilot 및 ROS 기반의 무인기(UAV)가 복잡한 환경에서 자율 주행(Waypoint Navigation), 다중 센서 융합 기반 장애물 회피(Hybrid Obstacle Avoidance), 그리고 비전 기반 정밀 착륙(Vision-based Precision Landing)을 수행하도록 설계된 통합 제어 프레임워크입니다.

주요 기능 및 알고리즘 (Key Features)
---
 1. 하이브리드 장애물 회피 (Hybrid Obstacle Avoidance)

Depth Camera (전역적 판단): 45도 형태의 거대 벽면을 인식하고 '우회 방향 잠금(Hysteresis Lock)'을 수행하여 연산 폭주(Time Jump) 및 갈팡질팡하는 진동 현상을 원천 차단합니다.
RPLiDAR + VFH (지역적 회피): 72방향으로 압축된 Vector Field Histogram을 통해 좁은 기둥이나 국소적 장애물을 부드럽게 회피합니다.

2. 시각 기반 정밀 착륙 (Vision-based Precision Landing)

 OpenCV & Kalman Filter: 하단 카메라를 통해 빨간색 랜딩 패드를 실시간으로 인식하며, 칼만 필터(Kalman Filter)를 통해 센서 노이즈를 제거하고 타겟의 위치를 부드럽게 추적합니다.
 IMU 자세 보상 (Attitude Compensation): 기체의 Roll/Pitch 기울어짐으로 인해 발생하는 카메라 프레임 상의 가짜 오차(Fake Error)를 물리·수학적으로 상쇄하여 정밀도를 극대화합니다.
 동적 하강 로직 (Dynamic Descent): 십자선 정렬 상태(Pixel Error)에 따라 하강 속도(Hovering -> 미세 하강 -> 정상 하강)를 동적으로 제어하여 패드를 벗어나는 추락을 방지합니다.

3. EKF2 고도 다중 센서 융합 (Altitude Sensor Fusion)

기압계(Barometer)의 환경적 드리프트를 극복하기 위해 1D LiDAR(Range Sensor) 데이터를 PX4 EKF2 알고리즘에 상시 융합합니다 
(`EKF2_HGT_REF = 2`, `EKF2_RNG_CTRL = 2`). 이를 통해 고속 강하 시에도 매우 정밀한 AGL(지상고) 추정치를 확보합니다.

---

시스템 아키텍처 및 노드 설명 (System Nodes)
---
본 패키지는 세 가지 주요 ROS 노드로 유기적으로 작동합니다.

1. `mission_controller.py` (비행 준비 및 제어 총괄)

 FCU 연결 상태를 확인하고 대회 규정 고도(2.5m)로 자동 이륙(Takeoff)합니다.

 안전 점검 후 `OFFBOARD` 모드로 진입하며, `path_follower` 노드로 제어권을 이양합니다.




2. `path_follower.py` (자율 주행 및 회피)

 CSV 파일(`practice_path.csv`)에 정의된 웨이포인트(x, y)를 Stanley Controller를 기반으로 추종합니다.

 주행 중 카메라가 랜딩 패드를 포착하거나 목적지에 도달하면 즉시 비행을 멈추고 `precision_lander`를 호출합니다.




3. `precision_lander.py` (정밀 착륙 제어)

 제어권을 인수받아 랜딩 패드 타겟 중앙에 위치를 동기화하며 하강합니다.


 목표 고도(0.3m) 및 오차(15px) 이내 도달 시 PX4 `AUTO.LAND` 모드를 트리거하여 안전하게 시동을 차단합니다.

---

실행 방법 (How to Run)
---
1. 사전 요구 사항 (Prerequisites)

* Ubuntu 20.04 & ROS Noetic
* PX4-Autopilot (SITL)
* Gazebo Classic

2. 패키지 실행 (Launch)

작성된 통합 런치 파일을 통해 `roscore`, Gazebo 시뮬레이터(`practice.world`), MAVROS, TF 변환, 그리고 3개의 제어 노드와 RViz가 한 번에 실행됩니다. 
`<master auto="start"/>` 태그가 포함되어 있어 ROS 마스터가 자동으로 실행 및 보장됩니다.

bash
# 워크스페이스 빌드 및 소싱 후 실행
roslaunch drone_practice [런치파일명].launch

---

실행 상세 정보:
* 사용 기체: LiDAR 및 Depth Camera가 장착된 `iris_rplidar` 모델이 스폰됩니다.

* TF 변환: `base_link` ->`rplidar_link`, `camera_link` 변환 및 `map` -> `odom` 변환이 백그라운드에서 자동 퍼블리시됩니다.

* 시각화: RViz 윈도우가 열리며 궤적, VFH 벡터, 로컬 카메라 디버깅 화면(`landing.rviz`)을 실시간으로 확인할 수 있습니다.

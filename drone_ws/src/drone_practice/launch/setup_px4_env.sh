#!/bin/bash
# PX4 경로
export PX4_DIR=$HOME/PX4-Autopilot

# 수정 1: 최신 PX4 버전의 setup_gazebo.bash 경로 반영
source $PX4_DIR/Tools/simulation/gazebo-classic/setup_gazebo.bash $PX4_DIR $PX4_DIR/build/px4_sitl_default

export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:$PX4_DIR
# 수정 2: 최신 PX4 버전의 sitl_gazebo 패키지 경로 반영
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:$PX4_DIR/Tools/simulation/gazebo-classic/sitl_gazebo-classic

# 우리 모델 경로 추가
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:$(rospack find drone_practice)/models

echo "PX4 환경 설정 완료"

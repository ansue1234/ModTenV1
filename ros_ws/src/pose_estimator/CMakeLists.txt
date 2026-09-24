cmake_minimum_required(VERSION 3.8)
project(pose_estimator)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate custom messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/IMUData.msg"
  "msg/MagnetometerData.msg"
  "msg/MagnetometerBias.msg"
  "msg/SequentialMagData.msg"
  "msg/SequentialMagArray.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

# Install Python executables
install(PROGRAMS
  scripts/imu_serial_node.py
  scripts/pose_estimator_node.py
  scripts/keyboard_publisher.py
  scripts/tendon_controller.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install the pretrained ONNX model (+ its .json contract)
install(DIRECTORY model
  DESTINATION share/${PROJECT_NAME}
)

# Install launch files
install(FILES
  launch/pose_estimator.launch.py
  launch/imu_serial.launch.py
  launch/tendon_system.launch.py
  DESTINATION share/${PROJECT_NAME}/launch/
)

# Install config files
install(FILES
  config/pose_estimator_params.yaml
  config/imu_params.yaml
  DESTINATION share/${PROJECT_NAME}/
)
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
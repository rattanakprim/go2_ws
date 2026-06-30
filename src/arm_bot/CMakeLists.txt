cmake_minimum_required(VERSION 3.8)
project(arm_bot)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

# The C++ Cartesian demos (move_program / cartesian_waypoint_demo) need MoveIt.
# Build them only when MoveIt is available so the Python kinematics nodes
# (fk/ik/nullspace_demo) still build on machines without a MoveIt install.
find_package(moveit_ros_planning_interface QUIET)
find_package(moveit_msgs QUIET)

install(
  DIRECTORY
    urdf
    launch
    config
    rviz
    meshes
  DESTINATION share/${PROJECT_NAME}/
)

if(moveit_ros_planning_interface_FOUND)
  add_executable(arm_bot src/move_program.cpp)
  ament_target_dependencies(arm_bot
    rclcpp
    geometry_msgs
    moveit_ros_planning_interface
    moveit_msgs
    visualization_msgs
    tf2_ros
  )

  add_executable(cartesian_waypoint_demo src/cartesian_waypoint_demo.cpp)
  ament_target_dependencies(cartesian_waypoint_demo
    rclcpp
    geometry_msgs
    moveit_ros_planning_interface
    moveit_msgs
    visualization_msgs
    tf2_ros
  )

  install(TARGETS
    arm_bot
    cartesian_waypoint_demo
    DESTINATION lib/${PROJECT_NAME}
  )
else()
  message(WARNING
    "moveit_ros_planning_interface not found — skipping the C++ Cartesian demos "
    "(arm_bot, cartesian_waypoint_demo). Python nodes still build.")
endif()

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python executables (authoritative kinematics + sim/hardware bridges)
install(PROGRAMS
  ${PROJECT_NAME}/fk_arm_final.py
  ${PROJECT_NAME}/ik_arm_final.py
  ${PROJECT_NAME}/ik_to_trajectory.py
  ${PROJECT_NAME}/relay_node.py
  ${PROJECT_NAME}/ik_verifier.py
  ${PROJECT_NAME}/joint_commands_bridge.py
  ${PROJECT_NAME}/cartesian_path.py
  ${PROJECT_NAME}/ee_trail_marker.py
  ${PROJECT_NAME}/send_test_goal.py
  ${PROJECT_NAME}/nullspace_demo.py
  ${PROJECT_NAME}/nullspace_rr.py
  DESTINATION lib/${PROJECT_NAME}
)

# ────────────────────────────────────────────────
# Testing / linting (good practice – optional but recommended)
# ────────────────────────────────────────────────
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# This must be the last line!
ament_package()

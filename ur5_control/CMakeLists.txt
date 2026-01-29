cmake_minimum_required(VERSION 3.8)
project(1425_ur5_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# Install Python scripts
install(PROGRAMS
    src/task5_perception.py
    DESTINATION lib/${PROJECT_NAME}
    RENAME task5_perception
)
install(PROGRAMS
    src/ebot_nav_task5.py
    DESTINATION lib/${PROJECT_NAME}
    RENAME ebot_nav_task5
)
install(PROGRAMS
    src/task5_manipulation.py
    DESTINATION lib/${PROJECT_NAME}
    RENAME task5_manipulation
)
install(PROGRAMS
    src/shape_detection_task5.py
    DESTINATION lib/${PROJECT_NAME}
    RENAME shape_detector_task5
)
ament_package()

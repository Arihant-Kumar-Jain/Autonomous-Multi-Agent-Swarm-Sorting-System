cmake_minimum_required(VERSION 3.5)
project(warehouse_multi_agent)

find_package(ament_cmake REQUIRED)

# Install Python scripts
install(PROGRAMS
  scripts/bfs_navigator.py
  scripts/rl_navigator.py
  scripts/task_coordinator.py
  scripts/metrics_logger.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

# Install world files
install(DIRECTORY
  worlds
  DESTINATION share/${PROJECT_NAME}
)

# Install models
install(DIRECTORY
  models
  DESTINATION share/${PROJECT_NAME}
)

# Install config
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()

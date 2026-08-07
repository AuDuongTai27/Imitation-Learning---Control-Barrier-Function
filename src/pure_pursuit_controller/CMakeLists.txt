cmake_minimum_required(VERSION 3.8)
project(pure_pursuit_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
# Install python modules
ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  # Controllers
  pure_pursuit_controller/controllers/controller_simulation.py
  pure_pursuit_controller/controllers/controller_real_run.py
  pure_pursuit_controller/controllers/ai_safe_drive_real.py
  pure_pursuit_controller/controllers/ai_inference_sim_override.py
  pure_pursuit_controller/controllers/claude_inference.py
  pure_pursuit_controller/controllers/mppi_basic.py
  pure_pursuit_controller/controllers/ai_safe_controller_jetson.py

  # AI Inference
  pure_pursuit_controller/ai_inference/ai_inference_sim.py
  pure_pursuit_controller/ai_inference/ai_inference_sim_odom.py
  pure_pursuit_controller/ai_inference/ai_inference_real.py
  pure_pursuit_controller/ai_inference/ai_inference_real_pytorch.py
  pure_pursuit_controller/ai_inference/dagger_inference_sim.py
  pure_pursuit_controller/ai_inference/dagger_inference_real_pytorch.py

  # Data Collection
  pure_pursuit_controller/data_collection/data_collection_sim.py
  pure_pursuit_controller/data_collection/data_collection_sim_20hz.py
  pure_pursuit_controller/data_collection/data_collection_sim_90.py
  pure_pursuit_controller/data_collection/data_collection_combined_sim.py
  pure_pursuit_controller/data_collection/data_collection_sim_90.py
  pure_pursuit_controller/data_collection/data_collection_real.py
  pure_pursuit_controller/data_collection/dagger_rrt_collector.py

  # CBF Safety Filter
  pure_pursuit_controller/cbf/cbf_safety_filter_node.py
  pure_pursuit_controller/cbf/cbf_core.py

  # Obstacle Spawners
  pure_pursuit_controller/obstacles/auto_obstacle_spawner.py
  pure_pursuit_controller/obstacles/obstacle_spawner.py

  # Training & Utilities
  pure_pursuit_controller/training/train.py
  pure_pursuit_controller/training/filter_dataset.py
  pure_pursuit_controller/training/trim_dataset.py
  pure_pursuit_controller/training/check_csv.py
  pure_pursuit_controller/training/eval_ai_performance.py
  pure_pursuit_controller/training/steering_logger.py
  pure_pursuit_controller/training/visualize_steering.py
  pure_pursuit_controller/training/downOnnx.py
  
  # scripts
  pure_pursuit_controller/scripts/sync_sim_pose.py
  pure_pursuit_controller/scripts/resolveWaypointPath.py

  # Benchmarking & Tools
  pure_pursuit_controller/tools/benchmark_frequency.py
  pure_pursuit_controller/tools/visualize_benchmark.py
  pure_pursuit_controller/tools/DO_HZ.py

  # RRT Planner
  rrt_planner/rrt.py
  rrt_planner/treeViz.py
  rrt_planner/utils.py
  rrt_planner/bsupline.py
  rrt_planner/rrt_node.py
  rrt_planner/hybrid_planner_sim.py
  rrt_planner/improved_hybrid_planner_sim.py
  rrt_planner/smart_pure_pursuit.py
  rrt_planner/code_chay_reak.py
  rrt_planner/rrt_expert_relay_node.py
  DESTINATION lib/${PROJECT_NAME})  

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

ament_package()

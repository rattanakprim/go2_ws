// cartesian_waypoint_demo.cpp
//
// Recreates the textbook "Cartesian waypoint planning" figures on the 7-DOF arm:
//   1. Move to the SRDF named state `bend` so the EE sits in a reachable
//      mid-workspace spot (this is waypoint 1 / center).
//   2. Define waypoint 2 (+Y, "right") and waypoint 3 (-Y, "left") as small
//      Cartesian offsets from waypoint 1.
//   3. Draw green spheres at the three waypoints, a red line strip for the
//      Cartesian path, and numbered text labels.
//   4. Execute straight-line Cartesian segments 1->2, 2->3, 3->1 with
//      MoveGroupInterface::computeCartesianPath (recreates Fig 3.48 / 3.49).
//   5. Enable "Show Trail" in RViz to get Fig 3.50.
//
// Run after `ros2 launch arm_moveit_config demo.launch.py` (move_group must be up),
// or use the convenience launch `cartesian_demo.launch.py`.

#include <cmath>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

using moveit::planning_interface::MoveGroupInterface;

static visualization_msgs::msg::Marker
makeSphere(const std::string &frame, int id, const geometry_msgs::msg::Pose &pose, double s)
{
  visualization_msgs::msg::Marker m;
  m.header.frame_id = frame;
  m.ns = "waypoints";
  m.id = id;
  m.type = visualization_msgs::msg::Marker::SPHERE;
  m.action = visualization_msgs::msg::Marker::ADD;
  m.pose = pose;
  m.pose.orientation.x = 0.0;  // markers ignore EE orientation; keep upright
  m.pose.orientation.y = 0.0;
  m.pose.orientation.z = 0.0;
  m.pose.orientation.w = 1.0;
  m.scale.x = s;
  m.scale.y = s;
  m.scale.z = s;
  m.color.r = 0.0f;
  m.color.g = 1.0f;  // green, like the figures
  m.color.b = 0.0f;
  m.color.a = 1.0f;
  return m;
}

static visualization_msgs::msg::Marker
makeText(const std::string &frame, int id, const geometry_msgs::msg::Pose &pose,
         const std::string &text, double h)
{
  visualization_msgs::msg::Marker m;
  m.header.frame_id = frame;
  m.ns = "labels";
  m.id = id;
  m.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
  m.action = visualization_msgs::msg::Marker::ADD;
  m.pose = pose;
  m.pose.position.z -= 0.04;  // label sits just below the sphere, like the figures
  m.pose.orientation.w = 1.0;
  m.scale.z = h;
  m.color.r = 1.0f;
  m.color.g = 1.0f;
  m.color.b = 1.0f;
  m.color.a = 1.0f;
  m.text = text;
  return m;
}

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("cartesian_waypoint_demo");
  auto logger = node->get_logger();

  // Parameters (tune for the arm's small reach; defaults are conservative).
  const double dy        = node->declare_parameter<double>("dy", 0.09);   // +/- sweep in Y (m)
  const double dx        = node->declare_parameter<double>("dx", 0.0);    // optional X offset (m)
  const double dz        = node->declare_parameter<double>("dz", 0.0);    // optional Z offset (m)
  const double eef_step  = node->declare_parameter<double>("eef_step", 0.005);
  const double sphere_sz = node->declare_parameter<double>("sphere_size", 0.02);
  const double vel_scale = node->declare_parameter<double>("vel_scale", 0.1);
  const std::string start_state  = node->declare_parameter<std::string>("start_state", "bend");
  const std::string marker_frame = node->declare_parameter<std::string>("marker_frame", "base_link");
  const std::string tip_link     = node->declare_parameter<std::string>("tip_link", "ee");
  const int    loops     = node->declare_parameter<int>("loops", 1);      // repeat the 1->2->3->1 cycle

  // MoveGroupInterface needs the node spinning to read the current state.
  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node);
  std::thread spinner([&exec]() { exec.spin(); });

  MoveGroupInterface mg(node, "arm");
  mg.setMaxVelocityScalingFactor(vel_scale);
  mg.setMaxAccelerationScalingFactor(vel_scale);
  mg.setPoseReferenceFrame(marker_frame);

  // Latched marker publisher so a late-joining RViz still sees the waypoints.
  rclcpp::QoS marker_qos(1);
  marker_qos.transient_local();
  auto marker_pub =
      node->create_publisher<visualization_msgs::msg::MarkerArray>("/cartesian_markers", marker_qos);

  // ---- 1. Move to the central "bend" configuration --------------------------
  RCLCPP_INFO(logger, "Moving to start state '%s'...", start_state.c_str());
  mg.setNamedTarget(start_state);
  MoveGroupInterface::Plan home_plan;
  if (!static_cast<bool>(mg.plan(home_plan))) {
    RCLCPP_ERROR(logger, "Failed to plan to '%s'. Aborting.", start_state.c_str());
    rclcpp::shutdown();
    if (spinner.joinable()) spinner.join();
    return 1;
  }
  mg.execute(home_plan);

  // ---- 2. Build the three Cartesian waypoints -------------------------------
  geometry_msgs::msg::Pose p1 = mg.getCurrentPose().pose;  // center (waypoint 1)
  geometry_msgs::msg::Pose p2 = p1;                        // right  (waypoint 2, +Y)
  p2.position.x += dx;
  p2.position.y += dy;
  p2.position.z += dz;
  geometry_msgs::msg::Pose p3 = p1;                        // left   (waypoint 3, -Y)
  p3.position.x -= dx;
  p3.position.y -= dy;
  p3.position.z -= dz;

  RCLCPP_INFO(logger, "Waypoint 1 (center): [%.3f, %.3f, %.3f]",
              p1.position.x, p1.position.y, p1.position.z);
  RCLCPP_INFO(logger, "Waypoint 2 (+Y)    : [%.3f, %.3f, %.3f]",
              p2.position.x, p2.position.y, p2.position.z);
  RCLCPP_INFO(logger, "Waypoint 3 (-Y)    : [%.3f, %.3f, %.3f]",
              p3.position.x, p3.position.y, p3.position.z);

  // ---- 3. Publish the markers (green spheres + red path line + labels) ------
  visualization_msgs::msg::MarkerArray markers;
  markers.markers.push_back(makeSphere(marker_frame, 1, p1, sphere_sz));
  markers.markers.push_back(makeSphere(marker_frame, 2, p2, sphere_sz));
  markers.markers.push_back(makeSphere(marker_frame, 3, p3, sphere_sz));
  markers.markers.push_back(makeText(marker_frame, 11, p1, "1", sphere_sz * 1.5));
  markers.markers.push_back(makeText(marker_frame, 12, p2, "2", sphere_sz * 1.5));
  markers.markers.push_back(makeText(marker_frame, 13, p3, "3", sphere_sz * 1.5));

  visualization_msgs::msg::Marker line;
  line.header.frame_id = marker_frame;
  line.ns = "path";
  line.id = 100;
  line.type = visualization_msgs::msg::Marker::LINE_STRIP;
  line.action = visualization_msgs::msg::Marker::ADD;
  line.scale.x = 0.004;  // line width
  line.color.r = 1.0f;   // red, like the figures
  line.color.a = 1.0f;
  line.pose.orientation.w = 1.0;
  line.points.push_back(p3.position);  // left -> center -> right
  line.points.push_back(p1.position);
  line.points.push_back(p2.position);
  markers.markers.push_back(line);

  marker_pub->publish(markers);
  RCLCPP_INFO(logger, "Published waypoint markers on /cartesian_markers");

  // ---- 3b. Live end-effector trail (TF base_link -> tip_link) ---------------
  // Sampled on a timer (driven by the background executor) while execute()
  // blocks the main thread, so the line grows as the EE actually moves.
  auto tf_buffer = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  auto tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

  rclcpp::QoS trail_qos(1);
  trail_qos.transient_local();
  auto trail_pub = node->create_publisher<visualization_msgs::msg::Marker>("/eef_trail", trail_qos);

  visualization_msgs::msg::Marker trail;
  trail.header.frame_id = marker_frame;
  trail.ns = "eef_trail";
  trail.id = 0;
  trail.type = visualization_msgs::msg::Marker::LINE_STRIP;
  trail.action = visualization_msgs::msg::Marker::ADD;
  trail.scale.x = 0.005;   // line width
  trail.color.r = 1.0f;    // yellow trail, distinct from the red waypoint path
  trail.color.g = 1.0f;
  trail.color.b = 0.0f;
  trail.color.a = 1.0f;
  trail.pose.orientation.w = 1.0;

  auto trail_timer = node->create_wall_timer(std::chrono::milliseconds(33), [&]() {
    geometry_msgs::msg::TransformStamped tf;
    try {
      tf = tf_buffer->lookupTransform(marker_frame, tip_link, tf2::TimePointZero);
    } catch (const tf2::TransformException &) {
      return;  // TF not ready yet
    }
    geometry_msgs::msg::Point p;
    p.x = tf.transform.translation.x;
    p.y = tf.transform.translation.y;
    p.z = tf.transform.translation.z;
    // Skip duplicate samples so the line only grows when the EE actually moves.
    if (!trail.points.empty()) {
      const auto &b = trail.points.back();
      double d2 = (p.x - b.x) * (p.x - b.x) + (p.y - b.y) * (p.y - b.y) + (p.z - b.z) * (p.z - b.z);
      if (d2 < 1e-8) return;
    }
    trail.points.push_back(p);
    trail.header.stamp = node->now();
    trail_pub->publish(trail);
  });

  // ---- 4. Execute straight-line Cartesian segments 1->2, 2->3, 3->1 ---------
  auto runSegment = [&](const geometry_msgs::msg::Pose &goal, const std::string &label) {
    std::vector<geometry_msgs::msg::Pose> wp{goal};
    moveit_msgs::msg::RobotTrajectory traj;
    double frac = mg.computeCartesianPath(wp, eef_step, 0.0, traj);
    RCLCPP_INFO(logger, "%s: Cartesian path %.1f%% achieved", label.c_str(), frac * 100.0);
    if (frac > 0.5) {
      MoveGroupInterface::Plan plan;
      plan.trajectory_ = traj;
      mg.execute(plan);
    } else {
      RCLCPP_WARN(logger, "%s: too little of the path is feasible (%.1f%%) - skipping. "
                          "Try a smaller 'dy'.", label.c_str(), frac * 100.0);
    }
  };

  for (int i = 0; i < loops && rclcpp::ok(); ++i) {
    RCLCPP_INFO(logger, "--- Cartesian cycle %d/%d ---", i + 1, loops);
    runSegment(p2, "1->2");
    runSegment(p3, "2->3");
    runSegment(p1, "3->1");
  }

  trail_timer->cancel();
  RCLCPP_INFO(logger, "Cartesian waypoint demo complete. EE trail has %zu points.",
              trail.points.size());
  rclcpp::shutdown();
  if (spinner.joinable()) spinner.join();
  return 0;
}

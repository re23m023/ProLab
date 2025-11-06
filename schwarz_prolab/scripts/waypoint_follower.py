#!/usr/bin/env python3
# =========================================================
# RUN 1 – Four-Waypoint Square Path (Simple Odometry Control)
# ---------------------------------------------------------
# - Drives through four fixed waypoints (square trajectory)
# - Uses odometry for feedback
# - Publishes /cmd_vel directly (no navigation stack)
# - Stops briefly at each corner
# =========================================================

import rospy
import math
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import tf

class WaypointFollowerRun1:
    def __init__(self):
        # --- Node setup ---
        rospy.init_node('waypoint_follower_run1', anonymous=True)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)

        # --- Define 4 waypoints (square path) ---
        self.waypoints = [
            (0.0,  3.0),   # ↑ forward
            (-3.0, 3.0),   # ← left
            (-3.0, -3.0),  # ↓ down
            (0.0, -3.0)    # → right
        ]

        # --- Parameters ---
        self.current_pose = (0.0, 0.0, 0.0)   # (x, y, yaw)
        self.goal_tolerance = 0.15            # [m]
        self.linear_speed = 0.25              # [m/s]
        self.angular_speed = 0.8              # [rad/s]
        self.rate = rospy.Rate(10)            # [Hz]

    # -----------------------------------------------------
    # Odometry callback
    # -----------------------------------------------------
    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        orientation_q = msg.pose.pose.orientation
        _, _, yaw = tf.transformations.euler_from_quaternion([
            orientation_q.x,
            orientation_q.y,
            orientation_q.z,
            orientation_q.w
        ])
        self.current_pose = (x, y, yaw)

    # -----------------------------------------------------
    # Normalize angle to [-pi, pi]
    # -----------------------------------------------------
    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    # -----------------------------------------------------
    # Rotate towards the waypoint
    # -----------------------------------------------------
    def turn_towards(self, goal_x, goal_y):
        while not rospy.is_shutdown():
            _, _, yaw = self.current_pose
            dx = goal_x - self.current_pose[0]
            dy = goal_y - self.current_pose[1]
            target_angle = math.atan2(dy, dx)
            angle_diff = self.normalize_angle(target_angle - yaw)

            if abs(angle_diff) < 0.05:
                break

            cmd = Twist()
            cmd.angular.z = self.angular_speed * angle_diff
            self.cmd_pub.publish(cmd)
            self.rate.sleep()
        self.stop_robot()

    # -----------------------------------------------------
    # Move straight toward the waypoint
    # -----------------------------------------------------
    def move_straight(self, goal_x, goal_y):
        while not rospy.is_shutdown():
            x, y, yaw = self.current_pose
            dx = goal_x - x
            dy = goal_y - y
            distance = math.sqrt(dx**2 + dy**2)

            if distance < self.goal_tolerance:
                break

            angle_to_goal = math.atan2(dy, dx)
            heading_error = self.normalize_angle(angle_to_goal - yaw)

            cmd = Twist()
            cmd.linear.x = self.linear_speed
            cmd.angular.z = 1.5 * heading_error  # proportional correction
            self.cmd_pub.publish(cmd)
            self.rate.sleep()
        self.stop_robot()

    # -----------------------------------------------------
    # Stop robot briefly
    # -----------------------------------------------------
    def stop_robot(self):
        cmd = Twist()
        self.cmd_pub.publish(cmd)
        rospy.sleep(1.0)

    # -----------------------------------------------------
    # Main routine
    # -----------------------------------------------------
    def run(self):
        rospy.sleep(2.0)  # wait for odom to stabilize
        rospy.loginfo("🚀 Starting Run 1 – Square trajectory")

        for i, (goal_x, goal_y) in enumerate(self.waypoints, start=1):
            rospy.loginfo(f"🧭 Waypoint {i}/{len(self.waypoints)}: ({goal_x:.2f}, {goal_y:.2f})")
            self.turn_towards(goal_x, goal_y)
            self.move_straight(goal_x, goal_y)
            rospy.loginfo(f"✅ Reached waypoint {i}")
            rospy.sleep(1.0)

        rospy.loginfo("🏁 Run 1 complete. Stopping robot and shutting down.")
        self.stop_robot()
        rospy.signal_shutdown("Run 1 finished")

# =========================================================
if __name__ == "__main__":
    try:
        node = WaypointFollowerRun1()
        node.run()
    except rospy.ROSInterruptException:
        pass


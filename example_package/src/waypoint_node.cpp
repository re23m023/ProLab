#include <ros/ros.h>
#include <actionlib/client/simple_action_client.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "waypoint_node");
    ros::NodeHandle nh("~");

    // Action Client für move_base
    actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> ac("move_base", true);

    ROS_INFO("Waiting for move_base action server to come up...");
    ac.waitForServer();
    ROS_INFO("Connected to move_base server.");

    // ---- Wegpunkte aus Parametern laden ----
    std::vector<double> wp1, wp2, wp3, wp4;
    if (!nh.getParam("waypoint1", wp1) ||
        !nh.getParam("waypoint2", wp2) ||
        !nh.getParam("waypoint3", wp3) ||
        !nh.getParam("waypoint4", wp4))
    {
        ROS_ERROR("Could not read waypoints from parameters!");
        return 1;
    }

    std::vector<std::vector<double>> waypoints = {wp1, wp2, wp3, wp4};

    ROS_INFO("Loaded %zu waypoints.", waypoints.size());

    // ---- Waypoints nacheinander abfahren ----
    for (size_t i = 0; i < waypoints.size(); ++i)
    {
        if (waypoints[i].size() < 3) {
            ROS_WARN("Waypoint %zu is incomplete (needs x,y,yaw). Skipping.", i+1);
            continue;
        }

        move_base_msgs::MoveBaseGoal goal;
        goal.target_pose.header.frame_id = "map";
        goal.target_pose.header.stamp = ros::Time::now();

        goal.target_pose.pose.position.x = waypoints[i][0];
        goal.target_pose.pose.position.y = waypoints[i][1];
        double yaw = waypoints[i][2];

        tf2::Quaternion q;
        q.setRPY(0, 0, yaw);
        goal.target_pose.pose.orientation = tf2::toMsg(q);

        ROS_INFO("Sending waypoint %zu: (x=%.2f, y=%.2f, yaw=%.2f rad)", 
                  i+1, waypoints[i][0], waypoints[i][1], yaw);

        ac.sendGoal(goal);

        ac.waitForResult(ros::Duration(120.0));  // Timeout optional (120 s pro Ziel)

        if (ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED)
            ROS_INFO("Reached waypoint %zu successfully.", i+1);
        else
            ROS_WARN("Failed to reach waypoint %zu (state: %s).", 
                     i+1, ac.getState().toString().c_str());
    }

    ROS_INFO("All waypoints completed.");
    return 0;
}

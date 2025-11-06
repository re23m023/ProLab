// ============================================================================
//  EXTENDED KALMAN FILTER
//  Uses: /odom (velocity) + /imu (yaw, omega_z)
//  Outputs: /prediction_EKF (PoseWithCovarianceStamped)
// ============================================================================

#include <ros/ros.h>
#include <eigen3/Eigen/Dense>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <cstdlib>   
#include <ctime>     // for std::time()

using Eigen::Matrix3d;
using Eigen::Vector3d;

// -----------------------------------------------------------------------------
// Utility: Normalize angle to range [-π, π]
// -----------------------------------------------------------------------------
static inline double normAngle(double a) {
    while (a >  M_PI) a -= 2.0 * M_PI;
    while (a < -M_PI) a += 2.0 * M_PI;
    return a;
}

// ============================================================================
//                         EKF NODE CLASS DEFINITION
// ============================================================================
class EKFNode {
public:
    // Constructor: initializes parameters, noise, ROS interfaces
    EKFNode(ros::NodeHandle& nh) : nh_(nh), initialized_(false)
    {
        std::srand(std::time(nullptr)); // Seed random generator (used for noise injection)

        // ----------------- Load ROS parameters with default values ----------
        nh_.param<std::string>("odom_topic", odom_topic_, "/odom");
        nh_.param<std::string>("imu_topic",  imu_topic_,  "/imu");
        nh_.param<std::string>("frame_id",   frame_id_,   "map");
        nh_.param<std::string>("prediction_topic", prediction_topic_, "/prediction_EKF");

        // ----------------- Define process noise covariance Q ---------------
        // Tuned for moderately noisy simulation (aligned with PF noise levels)
        Q_ = Matrix3d::Zero();
        Q_(0,0) = 0.015;   // x noise
        Q_(1,1) = 0.015;   // y noise
        Q_(2,2) = 0.008;   // yaw noise

        R_ = Matrix3d::Zero();
        R_(2,2) = 0.025;   // yaw measurement noise

        // ----------------- Initialize state and covariance -----------------
        x_.setZero();                          // Initial state: [x, y, theta]
        P_ = Matrix3d::Identity() * 0.1;       // Initial uncertainty

        // ----------------- Initialize ROS publisher ------------------------
        pub_ = nh_.advertise<geometry_msgs::PoseWithCovarianceStamped>(
                    prediction_topic_, 10);

        // ----------------- Set up time-synchronized subscribers ------------
        odom_sub_.subscribe(nh_, odom_topic_, 10);
        imu_sub_.subscribe(nh_,  imu_topic_,  10);
        sync_.reset(new message_filters::TimeSynchronizer<
                    nav_msgs::Odometry, sensor_msgs::Imu>(
                        odom_sub_, imu_sub_, 10));
        sync_->registerCallback(boost::bind(&EKFNode::callback, this, _1, _2));

        ROS_INFO("[EKF realistic PF-style] Running with tuned noise (mid precision).");
    }

private:
    // ------------------------------------------------------------------------
    // Internal state
    // ------------------------------------------------------------------------
    ros::NodeHandle nh_;
    ros::Publisher pub_;
    message_filters::Subscriber<nav_msgs::Odometry> odom_sub_;
    message_filters::Subscriber<sensor_msgs::Imu> imu_sub_;
    std::shared_ptr<
        message_filters::TimeSynchronizer<nav_msgs::Odometry, sensor_msgs::Imu>
    > sync_;

    std::string odom_topic_, imu_topic_, frame_id_, prediction_topic_;

    Vector3d x_;          // State vector: [x, y, theta]
    Matrix3d P_;          // Covariance matrix
    Matrix3d Q_, R_;      // Process and measurement noise
    ros::Time last_stamp_;
    bool initialized_;    // Flag to indicate whether the filter is initialized

    // =======================================================================
    // Main callback: synchronized odometry and IMU messages
    // =======================================================================
    void callback(const nav_msgs::Odometry::ConstPtr& odom,
                  const sensor_msgs::Imu::ConstPtr& imu)
    {
        ros::Time stamp = odom->header.stamp;

        // --------- First call: initialize state from sensors ---------------
        if (!initialized_) {
            x_(0) = odom->pose.pose.position.x;
            x_(1) = odom->pose.pose.position.y;
            x_(2) = tf2::getYaw(imu->orientation);
            last_stamp_ = stamp;
            initialized_ = true;
            publish(stamp);
            ROS_INFO("EKF initialized at (%.2f, %.2f, %.2f rad)",
                     x_(0), x_(1), x_(2));
            return;
        }

        // --------- Time delta calculation ----------------------------------
        double dt = (stamp - last_stamp_).toSec();
        if (dt <= 0.0) return;  // Skip if time difference is invalid
        last_stamp_ = stamp;

        // --------- Control inputs (v: linear velocity, w: angular velocity)
        double v = odom->twist.twist.linear.x;
        double w = imu->angular_velocity.z;

        // --------- Predict and correct steps -------------------------------
        predict(v, w, dt);
        correctYaw(tf2::getYaw(imu->orientation));

        publish(stamp);
    }

    // =======================================================================
    // EKF Predict step: non-linear motion model g(x,u)
    // =======================================================================
    void predict(double v, double w, double dt)
    {
        double th = x_(2); // current yaw

        Vector3d x_pred = x_;

        // ----- Motion model: differential drive robot ----------------------
        if (std::fabs(w) > 1e-6) {
            // Turning motion (w ≠ 0)
            double th_new = th + w * dt;
            x_pred(0) += (v / w) * (std::sin(th_new) - std::sin(th));
            x_pred(1) += (v / w) * (-std::cos(th_new) + std::cos(th));
            x_pred(2)  = normAngle(th_new);
        } else {
            // Straight-line motion (w ≈ 0)
            x_pred(0) += v * std::cos(th) * dt;
            x_pred(1) += v * std::sin(th) * dt;
            x_pred(2)  = normAngle(th + w * dt);  // Still add small angular change
        }

        // ----- Jacobian G: ∂g/∂x (linearization around current state) ------
        Matrix3d G = Matrix3d::Identity();
        G(0,2) = -v * std::sin(th) * dt;
        G(1,2) =  v * std::cos(th) * dt;

        // ----- Apply prediction update -------------------------------------
        x_ = x_pred;
        P_ = G * P_ * G.transpose() + Q_;
    }

    // =======================================================================
    // EKF Correction step: IMU yaw measurement update
    // =======================================================================
    void correctYaw(double yaw_meas)
    {
        // ----- Measurement model: z = h(x) = theta, so H = [0 0 1] ---------
        Eigen::RowVector3d H; H << 0, 0, 1;

        double z_pred = x_(2);                              // Predicted yaw
        double y = normAngle(yaw_meas - z_pred);            // Innovation


        // ----- Innovation covariance ---------------------------------------
        double S = H * P_ * H.transpose() + R_(2,2);

        // ----- Kalman gain (scaled by 90%) --------------------------------
        Eigen::Vector3d K = 0.9 * (P_ * H.transpose() * (1.0 / S));

        // ----- Correction -------------------------------------------------
        x_ += K * y;
        x_(2) = normAngle(x_(2));

        Matrix3d I = Matrix3d::Identity();
        P_ = (I - K * H) * P_;

        // ----- Limit yaw covariance to avoid divergence -------------------
        if (P_(2,2) > 0.25) P_(2,2) = 0.25;
    }

    // =======================================================================
    // Publishes current EKF pose estimate as ROS message
    // =======================================================================
    void publish(const ros::Time& stamp)
    {
        geometry_msgs::PoseWithCovarianceStamped msg;
        msg.header.stamp = stamp;
        msg.header.frame_id = frame_id_;

        msg.pose.pose.position.x = x_(0);
        msg.pose.pose.position.y = x_(1);

        tf2::Quaternion q;
        q.setRPY(0, 0, x_(2));  // Convert yaw to quaternion
        msg.pose.pose.orientation = tf2::toMsg(q);

        // Set 6x6 covariance matrix (flattened row-major)
        for (double &c : msg.pose.covariance) c = 0.0;
        msg.pose.covariance[0]  = P_(0,0);   // x variance
        msg.pose.covariance[7]  = P_(1,1);   // y variance
        msg.pose.covariance[35] = P_(2,2);   // yaw variance

        pub_.publish(msg);
    }
};

// ============================================================================
//  Main function
// ============================================================================
int main(int argc, char** argv)
{
    ros::init(argc, argv, "filter_node_EKF_realistic_PF_noise");
    ros::NodeHandle nh("~");      // Use private namespace (~) for parameter loading
    EKFNode node(nh);             // Create EKF node
    ros::spin();                  // Keep node alive
    return 0;
}

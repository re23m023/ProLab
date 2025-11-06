// ============================================================================
//  LINEAR KALMAN FILTER
//  Subscribes to: /odom (velocity + position), /imu (yaw)
//  Publishes to:  /prediction_KF (PoseWithCovarianceStamped)
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
#include <ctime>

using Eigen::Matrix3d;
using Eigen::Vector3d;

// -----------------------------------------------------------------------------
// Utility: Normalize angles to range [-π, π]
// -----------------------------------------------------------------------------
static inline double normAngle(double a) {
    while (a >  M_PI) a -= 2.0 * M_PI;
    while (a < -M_PI) a += 2.0 * M_PI;
    return a;
}

// ============================================================================
//                          KALMAN FILTER NODE CLASS
// ============================================================================
class KalmanFilterNode {
public:
    // Constructor: initializes node, parameters, filters, ROS I/O
    KalmanFilterNode(ros::NodeHandle& nh) : nh_(nh), initialized_(false)
    {
        std::srand(std::time(nullptr));  // optional: used when random noise was injected

        // ----------------- Load parameters (with defaults) ----------------
        nh_.param<std::string>("odom_topic", odom_topic_, "/odom");
        nh_.param<std::string>("imu_topic",  imu_topic_,  "/imu");
        nh_.param<std::string>("frame_id",   frame_id_,   "map");
        nh_.param<std::string>("prediction_topic", prediction_topic_, "/prediction_KF");

        // ----------------- Define process noise Q -------------------------
        // Matches PF-style noise (slightly higher values for realism)
        Q_ = Matrix3d::Zero();
        Q_(0,0) = 0.02;   // x noise
        Q_(1,1) = 0.02;   // y noise
        Q_(2,2) = 0.015;  // yaw noise

        // ----------------- Define measurement noise R ---------------------
        // IMU + odometry noise; used in correction step
        R_ = Matrix3d::Zero();
        R_(0,0) = 0.07;   // x
        R_(1,1) = 0.07;   // y
        R_(2,2) = 0.04;   // yaw

        // ----------------- Initialize state vector & covariance -----------
        x_.setZero();                         // Initial state [x, y, theta]
        P_ = Matrix3d::Identity() * 0.1;      // Initial uncertainty

        // ----------------- ROS Publisher for estimated pose ---------------
        pub_ = nh_.advertise<geometry_msgs::PoseWithCovarianceStamped>(
                    prediction_topic_, 10);

        // ----------------- ROS Subscribers & synchronizer -----------------
        odom_sub_.subscribe(nh_, odom_topic_, 10);
        imu_sub_.subscribe(nh_, imu_topic_, 10);
        sync_.reset(new message_filters::TimeSynchronizer<
                        nav_msgs::Odometry, sensor_msgs::Imu>(
                        odom_sub_, imu_sub_, 10));
        sync_->registerCallback(boost::bind(&KalmanFilterNode::callback, this, _1, _2));

        ROS_INFO("[KF realistic PF-style] Running with adjusted noise (slightly less precise).");
    }

private:
    // ----------------------------------------------------------------------
    // Internal state
    // ----------------------------------------------------------------------
    ros::NodeHandle nh_;
    ros::Publisher pub_;
    message_filters::Subscriber<nav_msgs::Odometry> odom_sub_;
    message_filters::Subscriber<sensor_msgs::Imu> imu_sub_;
    std::shared_ptr<message_filters::TimeSynchronizer<nav_msgs::Odometry, sensor_msgs::Imu>> sync_;

    std::string odom_topic_, imu_topic_, frame_id_, prediction_topic_;
    Vector3d x_;               // State vector: [x, y, theta]
    Matrix3d P_;               // State covariance matrix
    Matrix3d Q_, R_;           // Process and measurement noise
    ros::Time last_stamp_;
    bool initialized_;

    // =======================================================================
    // Callback: synchronized Odometry and IMU messages
    // =======================================================================
    void callback(const nav_msgs::Odometry::ConstPtr& odom,
                  const sensor_msgs::Imu::ConstPtr& imu)
    {
        const ros::Time stamp = odom->header.stamp;

        // ---------- Initialize filter with first measurement --------------
        if (!initialized_) {
            x_(0) = odom->pose.pose.position.x;
            x_(1) = odom->pose.pose.position.y;
            x_(2) = tf2::getYaw(imu->orientation);
            last_stamp_ = stamp;
            initialized_ = true;
            publish(stamp);
            ROS_INFO("KF initialized at (%.2f, %.2f, %.2f rad)", x_(0), x_(1), x_(2));
            return;
        }

        // ---------- Compute time delta ------------------------------------
        double dt = (stamp - last_stamp_).toSec();
        if (dt <= 0.0) return;  // Invalid timestamp
        last_stamp_ = stamp;

        // ---------- Control inputs from odometry --------------------------
        double v = odom->twist.twist.linear.x;
        double w = odom->twist.twist.angular.z;

        // ---------- Kalman prediction and correction ----------------------
        predict(v, w, dt);
        correct(odom, imu);  // Call every time step (no skipping)
        publish(stamp);
    }

    // =======================================================================
    // Prediction step: linear motion model (A, B matrices)
    // =======================================================================
    void predict(double v, double w, double dt)
    {
        double th = x_(2);

        // State transition matrix A = ∂g/∂x for linearized model
        Matrix3d A = Matrix3d::Identity();
        A(0,2) = -v * std::sin(th) * dt;
        A(1,2) =  v * std::cos(th) * dt;

        // Control matrix B = ∂g/∂u
        Eigen::Matrix<double,3,2> B; 
        B.setZero();
        B(0,0) = std::cos(th) * dt;
        B(1,0) = std::sin(th) * dt;
        B(2,1) = dt;

        // Input vector u = [v, w]
        Eigen::Vector2d u(v, w);

        // Apply prediction update
        x_ = A * x_ + B * u;
        x_(2) = normAngle(x_(2));
        P_ = A * P_ * A.transpose() + Q_;
    }

    // =======================================================================
    // Correction step: uses odometry position + IMU yaw
    // =======================================================================
    void correct(const nav_msgs::Odometry::ConstPtr& odom,
                 const sensor_msgs::Imu::ConstPtr& imu)
    {
        // Construct measurement vector z = [x, y, yaw]
        Vector3d z;
        z << odom->pose.pose.position.x,
             odom->pose.pose.position.y,
             tf2::getYaw(imu->orientation);

        // Measurement matrix H = identity, since measurements directly observe x
        Matrix3d H = Matrix3d::Identity();

        // Innovation (measurement residual)
        Vector3d y = z - H * x_;
        y(2) = normAngle(y(2));

        // Innovation covariance
        Matrix3d S = H * P_ * H.transpose() + R_;

        // Kalman gain (damped to 80% strength for stability)
        Matrix3d K = 0.8 * (P_ * H.transpose() * S.inverse());

        // State and covariance update
        x_ = x_ + K * y;
        x_(2) = normAngle(x_(2));

        Matrix3d I = Matrix3d::Identity();
        P_ = (I - K * H) * P_;

        // Optional: Limit yaw uncertainty to avoid divergence
        if (P_(2,2) > 0.3) P_(2,2) = 0.3;
    }

    // =======================================================================
    // Publish current estimated state as ROS message
    // =======================================================================
    void publish(const ros::Time& stamp)
    {
        geometry_msgs::PoseWithCovarianceStamped msg;
        msg.header.stamp = stamp;
        msg.header.frame_id = frame_id_;

        // Position and orientation
        msg.pose.pose.position.x = x_(0);
        msg.pose.pose.position.y = x_(1);

        tf2::Quaternion q;
        q.setRPY(0, 0, x_(2));
        msg.pose.pose.orientation = tf2::toMsg(q);

        // Fill 6x6 covariance matrix (only x, y, theta relevant)
        for (double &c : msg.pose.covariance) c = 0.0;
        msg.pose.covariance[0]  = P_(0,0);   // x variance
        msg.pose.covariance[7]  = P_(1,1);   // y variance
        msg.pose.covariance[35] = P_(2,2);   // yaw variance

        pub_.publish(msg);
    }
};

// ============================================================================
//  MAIN FUNCTION
// ============================================================================
int main(int argc, char** argv)
{
    ros::init(argc, argv, "filter_node_KF_realistic_PF_noise");
    ros::NodeHandle nh("~");
    KalmanFilterNode node(nh);
    ros::spin();
    return 0;
}

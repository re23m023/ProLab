#include <ros/ros.h>
#include <eigen3/Eigen/Dense>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

//////////////////////////////////////////////////////////
//  Extended Kalman Filter - integriert in den Node
//////////////////////////////////////////////////////////
class ExtendedKalmanFilter {
public:
    ExtendedKalmanFilter() {
        mu_ = Eigen::VectorXd::Zero(6);
        sigma_ = Eigen::MatrixXd::Identity(6, 6) * 0.1;

        Q_ = Eigen::MatrixXd::Identity(6, 6) * 0.01;
        R_ = Eigen::MatrixXd::Identity(3, 3) * 0.05;
    }

    void predict(const Eigen::VectorXd &u, double dt) {
        // u = [v_x, v_y, omega]
        Eigen::VectorXd mu_pred = mu_;
        mu_pred(0) += mu_(3) * dt;                         // x += vx * dt
        mu_pred(1) += mu_(4) * dt;                         // y += vy * dt
        mu_pred(2) = normalizeAngle(mu_(2) + mu_(5) * dt); // theta += omega * dt

        // Update velocities with control input
        mu_pred(3) = u(0);
        mu_pred(4) = u(1);
        mu_pred(5) = u(2);

        Eigen::MatrixXd F = Eigen::MatrixXd::Identity(6, 6);
        F(0,3) = dt;
        F(1,4) = dt;
        F(2,5) = dt;

        mu_ = mu_pred;
        sigma_ = F * sigma_ * F.transpose() + Q_;
    }

    void correct(const Eigen::VectorXd &z) {
        // z = [x_odom, y_odom, theta_imu]
        Eigen::VectorXd z_pred(3);
        z_pred << mu_(0), mu_(1), mu_(2);

        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3,6);
        H(0,0) = 1.0;
        H(1,1) = 1.0;
        H(2,2) = 1.0;

        Eigen::VectorXd y = z - z_pred;
        y(2) = normalizeAngle(y(2));

        Eigen::MatrixXd S = H * sigma_ * H.transpose() + R_;
        Eigen::MatrixXd K = sigma_ * H.transpose() * S.inverse();

        mu_ = mu_ + K * y;
        sigma_ = (Eigen::MatrixXd::Identity(6,6) - K * H) * sigma_;
    }

    Eigen::VectorXd getState() const { return mu_; }
    Eigen::MatrixXd getCovariance() const { return sigma_; }

private:
    Eigen::VectorXd mu_;
    Eigen::MatrixXd sigma_, Q_, R_;

    double normalizeAngle(double angle) const {
        while (angle > M_PI) angle -= 2.0 * M_PI;
        while (angle < -M_PI) angle += 2.0 * M_PI;
        return angle;
    }
};

//////////////////////////////////////////////////////////
//  ROS Node für EKF
//////////////////////////////////////////////////////////
class EKFFilterNode
{
public:
    EKFFilterNode(ros::NodeHandle &nh)
    {
        odom_sub_.subscribe(nh, "/odom", 10);
        imu_sub_.subscribe(nh, "/imu", 10);

        sync_.reset(new message_filters::TimeSynchronizer<nav_msgs::Odometry, sensor_msgs::Imu>(odom_sub_, imu_sub_, 10));
        sync_->registerCallback(boost::bind(&EKFFilterNode::sensorCallback, this, _1, _2));

        pub_ = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("/prediction", 10);

        ROS_INFO("EKF Filter Node initialized and listening to /odom and /imu");
    }

private:
    void sensorCallback(const nav_msgs::Odometry::ConstPtr &odom_msg,
                        const sensor_msgs::Imu::ConstPtr &imu_msg)
    {
        static bool first = true;
        static ros::Time last_time;

        if (first) {
            last_time = odom_msg->header.stamp;
            first = false;
            return;
        }

        double dt = (odom_msg->header.stamp - last_time).toSec();
        if (dt <= 0.0) return;
        last_time = odom_msg->header.stamp;

        // Control input (vx, vy, omega)
        double vx = odom_msg->twist.twist.linear.x;
        double vy = odom_msg->twist.twist.linear.y;
        double omega = odom_msg->twist.twist.angular.z;
        Eigen::VectorXd u(3);
        u << vx, vy, omega;

        ekf_.predict(u, dt);

        // Measurement (x, y from odom, theta from IMU)
        double x = odom_msg->pose.pose.position.x;
        double y = odom_msg->pose.pose.position.y;

        tf2::Quaternion q;
        tf2::fromMsg(imu_msg->orientation, q);
        double roll, pitch, yaw;
        tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);

        Eigen::VectorXd z(3);
        z << x, y, yaw;

        ekf_.correct(z);

        // Publish result
        geometry_msgs::PoseWithCovarianceStamped msg;
        msg.header.stamp = ros::Time::now();
        msg.header.frame_id = "odom";

        Eigen::VectorXd state = ekf_.getState();
        msg.pose.pose.position.x = state(0);
        msg.pose.pose.position.y = state(1);

        tf2::Quaternion q_out;
        q_out.setRPY(0, 0, state(2));
        msg.pose.pose.orientation = tf2::toMsg(q_out);

        Eigen::MatrixXd cov = ekf_.getCovariance();
        for (int i = 0; i < 36; ++i) msg.pose.covariance[i] = 0.0;
        msg.pose.covariance[0]  = cov(0,0);   // var(x)
        msg.pose.covariance[7]  = cov(1,1);   // var(y)
        msg.pose.covariance[35] = cov(2,2);   // var(theta)

        pub_.publish(msg);

        ROS_INFO_STREAM_THROTTLE(1.0, "EKF state -> x: " << state(0) << ", y: " << state(1)
                                  << ", theta: " << state(2) * 180.0 / M_PI << " deg");
    }

    ExtendedKalmanFilter ekf_;

    message_filters::Subscriber<nav_msgs::Odometry> odom_sub_;
    message_filters::Subscriber<sensor_msgs::Imu> imu_sub_;
    std::shared_ptr<message_filters::TimeSynchronizer<nav_msgs::Odometry, sensor_msgs::Imu>> sync_;
    ros::Publisher pub_;
};

//////////////////////////////////////////////////////////
//  Main
//////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    ros::init(argc, argv, "ekf_filter_node");
    ros::NodeHandle nh("~");
    EKFFilterNode node(nh);
    ros::spin();
    return 0;
}


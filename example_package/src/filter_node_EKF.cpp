#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <eigen3/Eigen/Dense>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

class EKFNode {
public:
    EKFNode(ros::NodeHandle& nh)
        : nh_(nh)
    {
        // === Parameter ===
        nh_.param<std::string>("odom_topic", odom_topic_, "/odom");
        nh_.param<std::string>("imu_topic", imu_topic_, "/imu");
        nh_.param<std::string>("frame_id", frame_id_, "map");
        nh_.param<std::string>("prediction_topic", prediction_topic_, "/prediction_EKF");

        std::vector<double> Qd{0.02,0.02,0.01}, Rd{0.10,0.10,0.05};
        nh_.getParam("Q_diag", Qd);
        nh_.getParam("R_diag", Rd);

        Q_ = Eigen::Vector3d(Qd[0], Qd[1], Qd[2]).asDiagonal();
        R_ = Eigen::Vector3d(Rd[0], Rd[1], Rd[2]).asDiagonal();

        x_.setZero(); // [x, y, theta]
        P_ = Eigen::Matrix3d::Identity() * 0.1;
        initialized_ = false;

        pub_ = nh_.advertise<geometry_msgs::PoseWithCovarianceStamped>(prediction_topic_, 10);

        odom_sub_.subscribe(nh_, odom_topic_, 10);
        imu_sub_.subscribe(nh_, imu_topic_, 10);
        sync_.reset(new message_filters::TimeSynchronizer<nav_msgs::Odometry, sensor_msgs::Imu>(odom_sub_, imu_sub_, 10));
        sync_->registerCallback(boost::bind(&EKFNode::sensorCallback, this, _1, _2));

        ROS_INFO_STREAM("Extended Kalman Filter Node running, publishing on " << prediction_topic_);
    }

private:
    ros::NodeHandle nh_;
    ros::Publisher pub_;
    message_filters::Subscriber<nav_msgs::Odometry> odom_sub_;
    message_filters::Subscriber<sensor_msgs::Imu> imu_sub_;
    std::shared_ptr< message_filters::TimeSynchronizer<nav_msgs::Odometry, sensor_msgs::Imu> > sync_;

    std::string odom_topic_, imu_topic_, frame_id_, prediction_topic_;
    Eigen::Vector3d x_;  // [x, y, theta]
    Eigen::Matrix3d P_, Q_, R_;
    bool initialized_;
    ros::Time last_stamp_;

    // --- Hilfsfunktion ---
    static double normAngle(double a) {
        while (a >  M_PI) a -= 2.0 * M_PI;
        while (a < -M_PI) a += 2.0 * M_PI;
        return a;
    }

    // --- Hauptcallback ---
    void sensorCallback(const nav_msgs::Odometry::ConstPtr& odom,
                        const sensor_msgs::Imu::ConstPtr& imu)
    {
        const ros::Time stamp = odom->header.stamp;
        if (!initialized_) {
            x_(0) = odom->pose.pose.position.x;
            x_(1) = odom->pose.pose.position.y;
            x_(2) = tf2::getYaw(imu->orientation);
            last_stamp_ = stamp;
            initialized_ = true;
            publish(stamp);
            return;
        }

        double dt = (stamp - last_stamp_).toSec();
        if (dt <= 0) return;
        last_stamp_ = stamp;

        double v = odom->twist.twist.linear.x;
        double w = odom->twist.twist.angular.z;

        predict(v, w, dt);

        Eigen::Vector3d z;
        z << odom->pose.pose.position.x,
             odom->pose.pose.position.y,
             tf2::getYaw(imu->orientation);

        correct(z);
        publish(stamp);
    }

    // --- PREDICT ---
    void predict(double v, double w, double dt)
    {
        double th = x_(2);

        // nichtlineares Bewegungsmodell
        x_(0) += v * std::cos(th) * dt;
        x_(1) += v * std::sin(th) * dt;
        x_(2) = normAngle(x_(2) + w * dt);

        // Jacobian F = d(g)/d(x)
        Eigen::Matrix3d F = Eigen::Matrix3d::Identity();
        F(0,2) = -v * std::sin(th) * dt;
        F(1,2) =  v * std::cos(th) * dt;

        // Kovarianzupdate
        P_ = F * P_ * F.transpose() + Q_;
    }

    // --- CORRECT ---
    void correct(const Eigen::Vector3d& z)
    {
        // Beobachtungsmodell: h(x) = x
        Eigen::Matrix3d H = Eigen::Matrix3d::Identity();

        Eigen::Vector3d y = z - x_;
        y(2) = normAngle(y(2));

        Eigen::Matrix3d S = H * P_ * H.transpose() + R_;
        Eigen::Matrix3d K = P_ * H.transpose() * S.inverse();

        x_ = x_ + K * y;
        x_(2) = normAngle(x_(2));

        Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
        P_ = (I - K * H) * P_;
    }

    // --- PUBLISH ---
    void publish(const ros::Time& stamp)
    {
        geometry_msgs::PoseWithCovarianceStamped msg;
        msg.header.stamp = stamp;
        msg.header.frame_id = frame_id_;

        msg.pose.pose.position.x = x_(0);
        msg.pose.pose.position.y = x_(1);
        msg.pose.pose.position.z = 0.0;

        tf2::Quaternion q;
        q.setRPY(0, 0, x_(2));
        msg.pose.pose.orientation = tf2::toMsg(q);

        for (double &c : msg.pose.covariance) c = 0.0;
        msg.pose.covariance[0]  = P_(0,0);
        msg.pose.covariance[7]  = P_(1,1);
        msg.pose.covariance[35] = P_(2,2);

        pub_.publish(msg);
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "filter_node_EKF");
    ros::NodeHandle nh("~");
    EKFNode node(nh);
    ros::spin();
    return 0;
}

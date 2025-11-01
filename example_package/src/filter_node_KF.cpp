#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Pose.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

#include <random>
#include <vector>
#include <cmath>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

// -------- PARTICLE STRUCT --------
struct Particle {
    double x;
    double y;
    double theta;
    double weight;
};

class ParticleFilterNode {
public:
    ParticleFilterNode(ros::NodeHandle& nh)
        : nh_(nh), gen_(std::random_device{}())
    {
        // === Parameter ===
        nh_.param<std::string>("odom_topic", odom_topic_, "/odom");
        nh_.param<std::string>("imu_topic", imu_topic_, "/imu");
        nh_.param<std::string>("frame_id", frame_id_, "map");
        nh_.param<std::string>("prediction_topic", prediction_topic_, "/prediction_particle");

        nh_.param<int>("num_particles", num_particles_, 300);
        nh_.param<double>("sigma_pos", sigma_pos_, 0.02);
        nh_.param<double>("sigma_theta", sigma_theta_, 0.02);

        // === Publisher ===
        pub_ = nh_.advertise<geometry_msgs::PoseArray>(prediction_topic_, 1);

        // === Subscribers ===
        odom_sub_.subscribe(nh_, odom_topic_, 10);
        imu_sub_.subscribe(nh_, imu_topic_, 10);
        sync_.reset(new message_filters::TimeSynchronizer<nav_msgs::Odometry, sensor_msgs::Imu>(odom_sub_, imu_sub_, 10));
        sync_->registerCallback(boost::bind(&ParticleFilterNode::sensorCallback, this, _1, _2));

        // === Init Particles ===
        initParticles();

        initialized_ = false;
        last_time_ = ros::Time::now();

        ROS_INFO("Particle Filter Node ready with %d particles", num_particles_);
    }

private:
    ros::NodeHandle nh_;
    ros::Publisher pub_;
    message_filters::Subscriber<nav_msgs::Odometry> odom_sub_;
    message_filters::Subscriber<sensor_msgs::Imu> imu_sub_;
    std::shared_ptr<message_filters::TimeSynchronizer<nav_msgs::Odometry, sensor_msgs::Imu>> sync_;

    std::string odom_topic_, imu_topic_, frame_id_, prediction_topic_;
    std::default_random_engine gen_;

    std::vector<Particle> particles_;
    int num_particles_;
    double sigma_pos_, sigma_theta_;
    bool initialized_;
    ros::Time last_time_;

    // === Hilfsfunktion ===
    static inline double normAngle(double a) {
        while (a >  M_PI) a -= 2.0 * M_PI;
        while (a < -M_PI) a += 2.0 * M_PI;
        return a;
    }

    // === Initialisierung ===
    void initParticles() {
        std::normal_distribution<double> npos(0.0, 0.05);
        std::normal_distribution<double> ntheta(0.0, 0.1);
        particles_.resize(num_particles_);

        for (auto &p : particles_) {
            p.x = npos(gen_);
            p.y = npos(gen_);
            p.theta = ntheta(gen_);
            p.weight = 1.0 / num_particles_;
        }
    }

    // === Hauptcallback ===
    void sensorCallback(const nav_msgs::Odometry::ConstPtr& odom,
                        const sensor_msgs::Imu::ConstPtr& imu)
    {
        ros::Time stamp = odom->header.stamp;
        if (!initialized_) {
            last_time_ = stamp;
            initialized_ = true;
            return;
        }

        double dt = (stamp - last_time_).toSec();
        if (dt <= 0) return;
        last_time_ = stamp;

        double v = odom->twist.twist.linear.x;
        double w = odom->twist.twist.angular.z;
        double yaw_meas = tf2::getYaw(imu->orientation);

        predict(v, w, dt);
        updateWeights(yaw_meas);
        resample();
        publish(stamp);
    }

    // === Prediction Step ===
    void predict(double v, double w, double dt)
    {
        std::normal_distribution<double> npos(0.0, sigma_pos_);
        std::normal_distribution<double> ntheta(0.0, sigma_theta_);

        for (auto &p : particles_) {
            p.x += v * std::cos(p.theta) * dt + npos(gen_);
            p.y += v * std::sin(p.theta) * dt + npos(gen_);
            p.theta = normAngle(p.theta + w * dt + ntheta(gen_));
        }
    }

    // === Update Step ===
    void updateWeights(double yaw_meas)
    {
        double sumw = 0.0;
        for (auto &p : particles_) {
            double diff = normAngle(yaw_meas - p.theta);
            // Gaussian likelihood (nur Orientierung wird hier genutzt)
            double w = std::exp(-0.5 * (diff * diff) / (sigma_theta_ * sigma_theta_));
            p.weight = w;
            sumw += w;
        }

        // Normalisieren
        for (auto &p : particles_) {
            p.weight /= (sumw + 1e-9);
        }
    }

    // === Resampling (Low-Variance Sampling) ===
    void resample()
    {
        std::vector<Particle> new_particles;
        new_particles.resize(num_particles_);

        double r = ((double)rand() / RAND_MAX) * (1.0 / num_particles_);
        double c = particles_[0].weight;
        int i = 0;

        for (int m = 0; m < num_particles_; ++m) {
            double U = r + (double)m / num_particles_;
            while (U > c && i < num_particles_ - 1) {
                i++;
                c += particles_[i].weight;
            }
            new_particles[m] = particles_[i];
            new_particles[m].weight = 1.0 / num_particles_;
        }

        particles_.swap(new_particles);
    }

    // === Publish Step ===
    void publish(const ros::Time& stamp)
    {
        geometry_msgs::PoseArray arr;
        arr.header.stamp = stamp;
        arr.header.frame_id = frame_id_;
        arr.poses.resize(num_particles_);

        for (size_t i = 0; i < particles_.size(); ++i) {
            geometry_msgs::Pose pose;
            pose.position.x = particles_[i].x;
            pose.position.y = particles_[i].y;
            pose.position.z = 0.0;
            tf2::Quaternion q;
            q.setRPY(0, 0, particles_[i].theta);
            pose.orientation = tf2::toMsg(q);
            arr.poses[i] = pose;
        }

        pub_.publish(arr);
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "filter_node_PF");
    ros::NodeHandle nh("~");
    ParticleFilterNode node(nh);
    ros::spin();
    return 0;
}

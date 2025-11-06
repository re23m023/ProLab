// ============================================================================
//  PARTICLE FILTER (Monte Carlo Localization)
//  Subscribes to: /cmd_vel (velocity), /imu (yaw), /odom /scan (laser)
//  Publishes to:  /prediction_particle (all particles), /prediction_pf_mean (mean pose)
//  Landmarks are detected using LaserScan, orientation is refined with IMU
// ============================================================================

#include <ros/ros.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/LaserScan.h>
#include <visualization_msgs/Marker.h>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <eigen3/Eigen/Dense>
#include <nav_msgs/Odometry.h>
#include <random>
#include <map>
#include <vector>
#include <numeric>
#include <cmath>

// ---------------------------------------------------------------------------
// Utility: Normalize angles to [-π, π]
// ---------------------------------------------------------------------------
static inline double normAngle(double a) {
    while (a >  M_PI) a -= 2.0 * M_PI;
    while (a < -M_PI) a += 2.0 * M_PI;
    return a;
}

// ---------------------------------------------------------------------------
// Particle structure representing a hypothesis of robot state
// ---------------------------------------------------------------------------
struct Particle {
    double x, y, theta;
    double weight;
};

// ============================================================================
//                            PARTICLE FILTER NODE CLASS
// ============================================================================
class ParticleFilterNode {
public:
    ParticleFilterNode(ros::NodeHandle& nh)
        : nh_(nh), gen_(std::random_device{}())  // random generator
    {
        // -------------------- Load Parameters from Launch File --------------------
        nh_.param<int>("num_particles", num_particles_, 500);
        nh_.param<std::string>("frame_id", frame_id_, "odom");
        nh_.param<std::string>("pub_topic", pub_topic_, "/prediction_pf_mean");

        // Motion noise parameters
        nh_.param<double>("alpha1", alpha1_, 0.01);
        nh_.param<double>("alpha2", alpha2_, 0.005);
        nh_.param<double>("alpha3", alpha3_, 0.005);
        nh_.param<double>("alpha4", alpha4_, 0.005);

        // Additional process noise (Gaussian)
        nh_.param<double>("sigma_x", sigma_x_, 0.01);
        nh_.param<double>("sigma_y", sigma_y_, 0.01);
        nh_.param<double>("sigma_theta", sigma_theta_, 0.02);

        // Measurement noise (used in weight update)
        nh_.param<double>("sigma_range", sigma_range_, 0.30);
        nh_.param<double>("sigma_yaw", sigma_yaw_, 0.12);

        // Roughening parameters (noise after resampling)
        nh_.param<double>("jitter_x", jitter_x_, 0.0);
        nh_.param<double>("jitter_y", jitter_y_, 0.0);
        nh_.param<double>("jitter_theta", jitter_theta_, 0.0);

        // Initial pose uncertainty
        nh_.param<double>("init_std_xy", init_std_xy_, 0.3);
        nh_.param<double>("init_std_th", init_std_th_, 0.3);

        // Outlier filtering (not actively used here)
        nh_.param<double>("outlier_distance_threshold", outlier_distance_threshold_, 2.0);
        nh_.param<int>("outlier_max_remove", outlier_max_remove_, 5);

        // Sign control for velocity and yaw rate (some robots need negated signs)
        nh_.param<double>("omega_sign", omega_sign_, 1.0);
        nh_.param<double>("v_sign", v_sign_, 1.0);

        // -------------------- Define known landmark positions ----------------------
        landmarks_ = {
            {1, Eigen::Vector2d( 2,  2)},  // red
            {2, Eigen::Vector2d(-2,  2)},  // green
            {3, Eigen::Vector2d( 2, -2)},  // blue
            {4, Eigen::Vector2d(-2, -2)}   // yellow
        };

        // -------------------- Initialize particles -------------------------------
        initParticles();

        // -------------------- Set up Subscribers -------------------------------
        cmd_sub_  = nh_.subscribe("/cmd_vel", 50, &ParticleFilterNode::cmdCallback, this);
        imu_sub_  = nh_.subscribe("/imu", 50, &ParticleFilterNode::imuCallback, this);
        scan_sub_ = nh_.subscribe("/scan", 10, &ParticleFilterNode::scanCallback, this);
        odom_sub_ = nh_.subscribe("/odom", 50, &ParticleFilterNode::odomCallback, this);

        // -------------------- Set up Publishers --------------------------------
        pub_particles_ = nh_.advertise<geometry_msgs::PoseArray>("/prediction_particle", 1);
        pub_mean_      = nh_.advertise<geometry_msgs::PoseWithCovarianceStamped>(pub_topic_, 1);
        debug_marker_  = nh_.advertise<visualization_msgs::Marker>("/pf_debug_landmark_hits", 1);

        last_time_ = ros::Time(0);
        imu_received_ = false;
        odom_received_ = false;
    }

private:
    // =========================================================================
    //                          INTERNAL STATE VARIABLES
    // =========================================================================
    ros::NodeHandle nh_;
    ros::Subscriber cmd_sub_, imu_sub_, scan_sub_, odom_sub_;
    ros::Publisher pub_particles_, pub_mean_, debug_marker_;

    ros::Time last_time_;                    // Last update time
    geometry_msgs::Twist last_cmd_;          // Last velocity command
    sensor_msgs::Imu last_imu_;              // Last IMU reading
    nav_msgs::Odometry last_odom_;           // Last odometry reading
    bool imu_received_, odom_received_;

    int num_particles_;
    std::string frame_id_, pub_topic_;
    std::vector<Particle> particles_;        // All particles
    std::map<int, Eigen::Vector2d> landmarks_; // Known map landmarks
    std::mt19937 gen_;                       // Random number generator

    // Noise parameters
    double alpha1_, alpha2_, alpha3_, alpha4_;
    double sigma_x_, sigma_y_, sigma_theta_;
    double sigma_range_, sigma_yaw_;
    double jitter_x_, jitter_y_, jitter_theta_;
    double outlier_distance_threshold_;
    int outlier_max_remove_;
    double omega_sign_, v_sign_;
    double init_std_xy_, init_std_th_;

    // =========================================================================
    //                          PARTICLE FILTER LOGIC
    // =========================================================================

    // --------------------------------------------
    // 1. Initialize particles around initial pose
    // --------------------------------------------
    void initParticles() {
        auto odom_msg = ros::topic::waitForMessage<nav_msgs::Odometry>("/odom", ros::Duration(5.0));
        // For debug
        if (!odom_msg) {
        ROS_WARN("Initial odometry not received! Particles may be off");
}


        double x0 = 0.0, y0 = 0.0, th0 = 0.0;
        if (odom_msg) {
            x0 = odom_msg->pose.pose.position.x;
            y0 = odom_msg->pose.pose.position.y;
            th0 = tf2::getYaw(odom_msg->pose.pose.orientation);
        }

        std::normal_distribution<double> dist_x(x0, init_std_xy_);
        std::normal_distribution<double> dist_y(y0, init_std_xy_);
        std::normal_distribution<double> dist_th(th0, init_std_th_);

        particles_.resize(num_particles_);
        for (auto& p : particles_) {
            p.x = dist_x(gen_);
            p.y = dist_y(gen_);
            p.theta = normAngle(dist_th(gen_));
            p.weight = 1.0 / num_particles_;
        }

        ROS_INFO("PF Initialized %d particles near (%.2f, %.2f)", num_particles_, x0, y0);
    }

    // --------------------------------------------
    // 2. Predict next state of each particle
    // --------------------------------------------
    void predict(double v_in, double w_in, double dt) {
        // Limit delta time to avoid instability
        dt = std::max(0.0, std::min(dt, 0.2));
        double v = v_sign_ * v_in;
        double w = omega_sign_ * w_in;

        std::normal_distribution<double> add_x(0.0, sigma_x_);
        std::normal_distribution<double> add_y(0.0, sigma_y_);
        std::normal_distribution<double> add_th(0.0, sigma_theta_);

        for (auto& p : particles_) {
            double v_hat = v + sampleNormal(alpha1_ * v * v + alpha2_ * w * w);
            double w_hat = w + sampleNormal(alpha3_ * v * v + alpha4_ * w * w);
            double gamma_hat = sampleNormal(alpha4_ * v * v + alpha1_ * w * w);

            if (std::fabs(w_hat) > 1e-6) {
                double th_new = p.theta + w_hat * dt;
                p.x += (-v_hat / w_hat) * sin(p.theta) + (v_hat / w_hat) * sin(th_new) + add_x(gen_);
                p.y += ( v_hat / w_hat) * cos(p.theta) - (v_hat / w_hat) * cos(th_new) + add_y(gen_);
                p.theta = normAngle(th_new + gamma_hat * dt + add_th(gen_));
            } else {
                p.x += v_hat * cos(p.theta) * dt + add_x(gen_);
                p.y += v_hat * sin(p.theta) * dt + add_y(gen_);
                p.theta = normAngle(p.theta + gamma_hat * dt + add_th(gen_));
            }
        }
    }

    // --------------------------------------------
    // 3. Update weights using LaserScan + IMU
    // --------------------------------------------
    void updateWeights(const sensor_msgs::LaserScan::ConstPtr& scan_msg) {
        if (!imu_received_ || !scan_msg) return;

        const double yaw_meas = tf2::getYaw(last_imu_.orientation);
        const double inv_sigma2_yaw = 1.0 / (sigma_yaw_ * sigma_yaw_);
        const double inv_sigma2_r   = 1.0 / (sigma_range_ * sigma_range_);
        const double eps = 1e-12;

        const int n = scan_msg->ranges.size();
        if (n == 0) return;

        // Select 64 beams evenly spaced across full scan
        std::vector<int> beam_indices;
        beam_indices.reserve(64);
        for (int i = 0; i < 64; ++i)
            beam_indices.push_back((i * n) / 64);

        double sum_linear = 0.0;
        std::vector<geometry_msgs::Point> debug_points;

        for (auto& p : particles_) {
            const double dth = normAngle(p.theta - yaw_meas);
            double log_w = -0.5 * dth * dth * inv_sigma2_yaw;  // IMU yaw likelihood
            int valid_beams = 0;

            // For each selected beam
            for (int idx : beam_indices) {
                const double meas_r = scan_msg->ranges[idx];
                if (!std::isfinite(meas_r) || meas_r < 0.05 || meas_r > 6.0) continue;
                ++valid_beams;

                // Transform scan endpoint into world coordinates
                const double beam_angle = scan_msg->angle_min + idx * scan_msg->angle_increment;
                const double beam_global = normAngle(p.theta + beam_angle);
                const double lx = p.x + meas_r * std::cos(beam_global);
                const double ly = p.y + meas_r * std::sin(beam_global);

                // Compute likelihood of scan point hitting a landmark surface
                const double lm_radius = 0.25;
                double best_like = eps;
                for (const auto& kv : landmarks_) {
                    const auto& lm = kv.second;
                    const double dx = lm(0) - lx;
                    const double dy = lm(1) - ly;
                    const double dist = std::sqrt(dx * dx + dy * dy);
                    double dist_c = std::max(0.0, dist - lm_radius);  // surface match
                    const double like = std::exp(-0.5 * dist_c * dist_c * inv_sigma2_r);
                    if (like > best_like) best_like = like;
                }
                log_w += std::log(best_like + eps);  // accumulate likelihood

                // Store point for visualization
                geometry_msgs::Point pt;
                pt.x = lx; pt.y = ly; pt.z = 0.05;
                debug_points.push_back(pt);
            }

            // Normalize weight by number of valid beams
            if (valid_beams > 0)
                log_w /= static_cast<double>(valid_beams);

            // Add a small odometry-based correction (optional)
            if (odom_received_) {
                const double x_o = last_odom_.pose.pose.position.x;
                const double y_o = last_odom_.pose.pose.position.y;
                const double sigma_odom = 0.3;
                const double inv_s2_odom = 1.0 / (sigma_odom * sigma_odom);
                const double dx = p.x - x_o;
                const double dy = p.y - y_o;
                const double w_odom = std::exp(-0.5 * (dx * dx + dy * dy) * inv_s2_odom);
                log_w += 0.1 * std::log(w_odom + eps);  // scaled down
            }

            p.weight = std::exp(log_w);
            sum_linear += p.weight;
        }

        // Normalize weights across all particles
        if (sum_linear <= eps) {
            const double w = 1.0 / static_cast<double>(num_particles_);
            for (auto& p : particles_) p.weight = w;
            return;
        }
        for (auto& p : particles_) p.weight /= sum_linear;

        // Compute effective number of particles (neff)
        double neff_inv = 0.0;
        for (const auto& p : particles_) neff_inv += p.weight * p.weight;
        const double neff = 1.0 / std::max(neff_inv, eps);
        if (neff < 0.8 * num_particles_) resample();  // low diversity → resample

        // Publish debug visualization of scan points
        visualization_msgs::Marker marker;
        marker.header.frame_id = frame_id_;
        marker.header.stamp = ros::Time::now();
        marker.ns = "pf_debug";
        marker.id = 0;
        marker.type = visualization_msgs::Marker::POINTS;
        marker.scale.x = 0.05;
        marker.scale.y = 0.05;
        marker.color.r = 0.2;
        marker.color.g = 1.0;
        marker.color.b = 0.2;
        marker.color.a = 0.9;
        marker.points = debug_points;
        debug_marker_.publish(marker);
    }

    // --------------------------------------------
    // 4. Resampling step (Multinomial Sampling)
    // --------------------------------------------
    void resample() {
        std::vector<Particle> new_particles;
        new_particles.reserve(num_particles_);
        std::vector<double> weights;
        for (const auto& p : particles_) weights.push_back(p.weight);

        std::discrete_distribution<int> dist(weights.begin(), weights.end());
        for (int i = 0; i < num_particles_; ++i) {
            new_particles.push_back(particles_[dist(gen_)]);
            new_particles.back().weight = 1.0 / num_particles_;
        }
        particles_.swap(new_particles);

        // Roughening (optional) – prevents particle collapse
        double mean_x = 0.0, mean_y = 0.0;
        for (auto& p : particles_) {
            mean_x += p.x;
            mean_y += p.y;
        }
        mean_x /= particles_.size();
        mean_y /= particles_.size();

        double var_x = 0.0, var_y = 0.0;
        for (auto& p : particles_) {
            var_x += (p.x - mean_x) * (p.x - mean_x);
            var_y += (p.y - mean_y) * (p.y - mean_y);
        }
        var_x /= particles_.size();
        var_y /= particles_.size();

        double rough_x = 0.05 * std::sqrt(var_x + 1e-6);
        double rough_y = 0.05 * std::sqrt(var_y + 1e-6);
        double rough_th = jitter_theta_ > 0 ? jitter_theta_ : 0.005;

        std::normal_distribution<double> jx(0.0, rough_x);
        std::normal_distribution<double> jy(0.0, rough_y);
        std::normal_distribution<double> jt(0.0, rough_th);

        for (auto& p : particles_) {
            p.x += jx(gen_);
            p.y += jy(gen_);
            p.theta = normAngle(p.theta + jt(gen_));
        }
    }

    // --------------------------------------------
    // 5. Publish particle cloud and mean estimate
    // --------------------------------------------
    void publish(const ros::Time& stamp) {
        geometry_msgs::PoseArray arr;
        arr.header.stamp = stamp;
        arr.header.frame_id = frame_id_;
        arr.poses.reserve(particles_.size());

        double sum_x = 0, sum_y = 0, sum_sin = 0, sum_cos = 0;
        for (auto& p : particles_) {
            geometry_msgs::Pose pose;
            pose.position.x = p.x;
            pose.position.y = p.y;
            tf2::Quaternion q;
            q.setRPY(0, 0, p.theta);
            pose.orientation = tf2::toMsg(q);
            arr.poses.push_back(pose);

            sum_x += p.x;
            sum_y += p.y;
            sum_cos += std::cos(p.theta);
            sum_sin += std::sin(p.theta);
        }
        pub_particles_.publish(arr);

        // Mean estimate
        double mean_x = sum_x / num_particles_;
        double mean_y = sum_y / num_particles_;
        double mean_th = std::atan2(sum_sin, sum_cos);

        // Estimate covariance (used by RViz display)
        double cxx = 0, cyy = 0, ctt = 0;
        for (auto& p : particles_) {
            cxx += (p.x - mean_x) * (p.x - mean_x);
            cyy += (p.y - mean_y) * (p.y - mean_y);
            double dth = normAngle(p.theta - mean_th);
            ctt += dth * dth;
        }
        cxx /= num_particles_;
        cyy /= num_particles_;
        ctt /= num_particles_;

        geometry_msgs::PoseWithCovarianceStamped msg;
        msg.header.stamp = stamp;
        msg.header.frame_id = frame_id_;
        msg.pose.pose.position.x = mean_x;
        msg.pose.pose.position.y = mean_y;
        tf2::Quaternion q;
        q.setRPY(0, 0, mean_th);
        msg.pose.pose.orientation = tf2::toMsg(q);
        msg.pose.covariance[0] = cxx;
        msg.pose.covariance[7] = cyy;
        msg.pose.covariance[35] = ctt;
        pub_mean_.publish(msg);
    }

    // --------------------------------------------
    // 6. Callbacks
    // --------------------------------------------
    void cmdCallback(const geometry_msgs::Twist::ConstPtr& msg) { last_cmd_ = *msg; }
    void imuCallback(const sensor_msgs::Imu::ConstPtr& msg) { last_imu_ = *msg; imu_received_ = true; }
    void odomCallback(const nav_msgs::Odometry::ConstPtr& msg) { last_odom_ = *msg; odom_received_ = true; }

    // Scan callback triggers a full PF update (predict + correct + publish)
    void scanCallback(const sensor_msgs::LaserScan::ConstPtr& msg) {
        ros::Time now = msg ? msg->header.stamp : ros::Time::now();
        if (last_time_.isZero()) { last_time_ = now; return; }

        double dt = (now - last_time_).toSec();
        last_time_ = now;
        if (dt <= 0.0 || dt > 0.5) return;

        const double v = last_cmd_.linear.x;
        const double w = last_cmd_.angular.z;

        predict(v, w, dt);
        updateWeights(msg);
        publish(now);
    }

    // --------------------------------------------
    // 7. Gaussian Sampling Helper
    // --------------------------------------------
    double sampleNormal(double var) {
        if (var <= 0.0) return 0.0;
        std::normal_distribution<double> d(0.0, std::sqrt(var));
        return d(gen_);
    }
};


// ============================================================================
// Main entry point
// ============================================================================
int main(int argc, char** argv) {
    ros::init(argc, argv, "filter_node_PF");
    ros::NodeHandle nh("~");
    ParticleFilterNode node(nh);
    ros::spin();
    return 0;
}

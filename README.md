start.launch

Starts the full Gazebo simulation with playground.world.

Spawns TurtleBot3 and starts all bringup nodes.

Launches alllocalization filters (KF, EKF, PF).

Opens RViz for visualization and enables keyboard teleop.


data_generator.launch

Runs a Python script with 4 waypoint followers that drives the robot through predefined points.

Simultaneously records all relevant topics to run1.bag (sensor data, odometry, filter outputs, etc.).


data_generator3.launch

Spawns 16 circular landmarks in Gazebo for a clockwise run.

Loads a shared URDF marker description for visualization.

Records the full dataset to run3.bag for later analysis.


KF.launch

Starts Gazebo, TurtleBot, and only the Kalman Filter node.



EKF.launch

Same setup


PF.launch

I think you know what to expect here

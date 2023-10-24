import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np
from numpy.linalg import inv

class KalmanFilter(Node):
    def __init__(self):
        super().__init__('kalman_filter_node')
        
        # Initialize kalman variables
        self.x_est = np.zeros((2, 1))  # Initial estimate of the state [x, y]
        self.P_est = np.array([[10, 0], [0, 10]])    # Initial estimate of the error covariance matrix
        self.Q = np.array([[0.1, 0], [0, 0.1]])    # Process noise covariance
        self.R = np.array([[0.01, 0], [0, 0.01]])        # Measurement noise covariance

        # Subscribe to the /odom_noise topic
        self.subscription = self.create_subscription(Odometry,
                                                     '/odom_noise',
                                                     self.odom_callback,
                                                     1)

        # Publish the estimated reading
        self.estimated_pub = self.create_publisher(Odometry, "/odom_estimated", 1)

        # Subscribe to the control input topic (e.g., /cmd_vel)
        self.control_input_subscription = self.create_subscription(Twist, '/cmd_vel', self.control_input_callback, 1)

        self.linear_velocity = 0
        self.angular_velocity = 0

    def control_input_callback(self, msg):
         self.linear_velocity = msg.linear.x
         self.angular_velocity = msg.angular.z

    def odom_callback(self, msg):
        # Extract the position measurements from the Odometry message
        z = np.array([[msg.pose.pose.position.x], [msg.pose.pose.position.y]])

        # Prediction step
        A = np.array([[1, 0], [0, 1]])  # State transition matrix
        B = np.array([[1, 0], [0, 1]])
        u = np.array([[self.linear_velocity], [self.angular_velocity]])

        # Predict the new state and error covariance
        x_pred = A @ self.x_est + B @ u
        P_pred = A @ self.P_est @ A.T + self.Q

        # Update step (Kalman Gain and Update the state estimate)
        H = np.identity(2)
        K = P_pred @ H.T @ inv(H @ P_pred @ H.T + self.R)  # Kalman Gain
        self.x_est = x_pred + K @ (z - H @ x_pred)
        self.P_est = (np.identity(2) - K @ H) @ P_pred

        # Create an Odometry message with the estimated position
        estimated_msg = Odometry()
        estimated_msg.pose.pose.position.x = self.x_est[0, 0]
        estimated_msg.pose.pose.position.y = self.x_est[1, 0]

        # Publish the estimated reading
        self.estimated_pub.publish(estimated_msg)

def main(args=None):
    try:
         rclpy.init(args=args)
         node = KalmanFilter()
         rclpy.spin(node)
         rclpy.shutdown()
    except KeyboardInterrupt:
         pass

if __name__ == '__main__':
        main()



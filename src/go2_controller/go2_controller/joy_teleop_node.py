"""
Joystick teleop for the Go2.

Reads /joy (from the `joy` driver node) and publishes:
  /cmd_vel    (geometry_msgs/Twist)  -- left stick = move, right stick = turn
  /go2/action (std_msgs/String)      -- buttons = jump / sit / stand

Default mapping (Xbox-style pad; remap with parameters if yours differs):
  axis 1 (left stick Y)  -> forward / back
  axis 0 (left stick X)  -> strafe left / right
  axis 3 (right stick X) -> turn left / right
  button 0 (A) -> jump   button 1 (B) -> sit   button 2 (X) -> stand

Run (with a controller plugged in):
  ros2 run joy joy_node
  ros2 run go2_controller joy_teleop
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
from std_msgs.msg import String


class Go2JoyTeleop(Node):
    def __init__(self):
        super().__init__("go2_joy_teleop")

        # axis / button indices (parameters so users can remap)
        self.declare_parameter("axis_forward", 1)
        self.declare_parameter("axis_strafe", 0)
        self.declare_parameter("axis_turn", 3)
        self.declare_parameter("btn_jump", 0)
        self.declare_parameter("btn_sit", 1)
        self.declare_parameter("btn_stand", 2)
        self.declare_parameter("max_lin", 0.4)     # m/s
        self.declare_parameter("max_ang", 1.0)     # rad/s
        self.declare_parameter("deadzone", 0.1)

        g = self.get_parameter
        self.ax_fwd = g("axis_forward").value
        self.ax_str = g("axis_strafe").value
        self.ax_turn = g("axis_turn").value
        self.b_jump = g("btn_jump").value
        self.b_sit = g("btn_sit").value
        self.b_stand = g("btn_stand").value
        self.max_lin = g("max_lin").value
        self.max_ang = g("max_ang").value
        self.deadzone = g("deadzone").value

        self.cmd_pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.action_pub = self.create_publisher(String, "go2/action", 10)
        self.create_subscription(Joy, "joy", self.on_joy, 10)

        self.last_buttons = []
        self.last_cmd = Twist()
        # Republish the last command at 20 Hz so the controller's safety
        # timeout never trips between joystick updates.
        self.create_timer(0.05, lambda: self.cmd_pub.publish(self.last_cmd))
        self.get_logger().info("Joy teleop ready (waiting for /joy).")

    def _dz(self, v):
        return 0.0 if abs(v) < self.deadzone else v

    def _axis(self, axes, i):
        return self._dz(axes[i]) if 0 <= i < len(axes) else 0.0

    def _pressed(self, buttons, i):
        """True only on the rising edge (just pressed this message)."""
        now = buttons[i] if 0 <= i < len(buttons) else 0
        was = self.last_buttons[i] if i < len(self.last_buttons) else 0
        return now and not was

    def on_joy(self, msg: Joy):
        cmd = Twist()
        cmd.linear.x = self._axis(msg.axes, self.ax_fwd) * self.max_lin
        cmd.linear.y = self._axis(msg.axes, self.ax_str) * self.max_lin
        cmd.angular.z = self._axis(msg.axes, self.ax_turn) * self.max_ang
        self.last_cmd = cmd
        self.cmd_pub.publish(cmd)

        for btn, action in ((self.b_jump, "jump"),
                            (self.b_sit, "sit"),
                            (self.b_stand, "stand")):
            if self._pressed(msg.buttons, btn):
                self.action_pub.publish(String(data=action))
                self.get_logger().info(f"action: {action}")
        self.last_buttons = list(msg.buttons)


def main(args=None):
    rclpy.init(args=args)
    node = Go2JoyTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

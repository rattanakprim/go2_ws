"""Color-blob tracker -- a reliable drop-in replacement for coco_detector in sim.

The MuJoCo-rendered ball isn't photorealistic, so MobileNet mis-labels it (sports ball /
frisbee / mouse / airplane depending on framing). Since we control the sim, the ball is a
known vivid colour, so a simple colour threshold tracks it 100% reliably at any distance --
no GPU, no class roulette.

Subscribes /camera/image_raw (rgb8), segments the magenta ball by an RGB threshold, and
publishes the SAME message the follower consumes: vision_msgs/Detection2DArray on
/detected_objects (class_id "ball", bbox = the blob's bounding box). So follow_node, the
ball-driving, and obstacle avoidance all work unchanged -- only the detector swaps.

Run:  ros2 run go2_controller color_tracker
"""
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import (Detection2DArray, Detection2D,
                             ObjectHypothesisWithPose)


class ColorTracker(Node):
    def __init__(self):
        super().__init__("color_tracker")
        # Magenta ball (rgba 0.95 0.05 0.80): high R, low G, high B. Thresholds are on
        # 0-255 and lighting-robust (use R-G / B-G margins so shading doesn't matter).
        self.declare_parameter("min_pixels", 60)     # ignore tiny specks / noise
        self.declare_parameter("rg_margin", 45)      # R-G and B-G must exceed this
        self.declare_parameter("min_rb", 70)         # R and B must be at least this bright
        self.declare_parameter("class_id", "ball")
        self.min_pixels = int(self.get_parameter("min_pixels").value)
        self.rg_margin = int(self.get_parameter("rg_margin").value)
        self.min_rb = int(self.get_parameter("min_rb").value)
        self.class_id = self.get_parameter("class_id").value

        self.pub = self.create_publisher(Detection2DArray, "detected_objects", 10)
        self.create_subscription(Image, "camera/image_raw", self._on_image, 5)
        self.get_logger().info("Colour tracker up (magenta ball -> /detected_objects).")
        self._logged = False

    def _on_image(self, msg):
        if msg.encoding != "rgb8":
            return
        if len(msg.data) != msg.height * msg.width * 3:   # guard partial/garbled frames
            return
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3).astype(np.int16)
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        mask = ((r - g > self.rg_margin) & (b - g > self.rg_margin) &
                (r > self.min_rb) & (b > self.min_rb))

        out = Detection2DArray()
        out.header = msg.header
        rows, cols = np.nonzero(mask)
        if rows.size >= self.min_pixels:
            x0, x1 = int(cols.min()), int(cols.max())
            y0, y1 = int(rows.min()), int(rows.max())
            d = Detection2D()
            d.header = msg.header
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = self.class_id
            hyp.hypothesis.score = 1.0
            d.results.append(hyp)
            d.bbox.center.position.x = float((x0 + x1) / 2.0)
            d.bbox.center.position.y = float((y0 + y1) / 2.0)
            d.bbox.size_x = float(x1 - x0 + 1)
            d.bbox.size_y = float(y1 - y0 + 1)
            out.detections.append(d)
            if not self._logged:
                self._logged = True
                self.get_logger().info("First ball blob tracked.")
        self.pub.publish(out)


def main():
    rclpy.init()
    node = ColorTracker()
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

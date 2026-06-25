import os
from glob import glob
from setuptools import find_packages, setup

package_name = "go2_controller"


def tree_data_files(top):
    """Install every file under <top>/ into share/<pkg>/<top>/, keeping layout."""
    entries = []
    for root, dirs, files in os.walk(top):
        dirs[:] = [d for d in dirs if d != ".git"]   # skip git internals
        files = [f for f in files if not f.startswith(".")]
        if not files:
            continue
        dest = os.path.join("share", package_name, root)
        entries.append((dest, [os.path.join(root, f) for f in files]))
    return entries


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ] + tree_data_files("models") + tree_data_files("web") + tree_data_files("config"),
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="nak",
    maintainer_email="orotha37@gmail.com",
    description="Velocity/teleop controller for the Unitree Go2 (MuJoCo sim).",
    license="MIT",
    entry_points={
        "console_scripts": [
            "teleop_controller = go2_controller.teleop_node:main",
            "gui_teleop = go2_controller.gui_teleop_node:main",
            "joy_teleop = go2_controller.joy_teleop_node:main",
            "explore = go2_controller.explore_node:main",
            "follow = go2_controller.follow_node:main",
            "color_tracker = go2_controller.color_tracker_node:main",
        ],
    },
)

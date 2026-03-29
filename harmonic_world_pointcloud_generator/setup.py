from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'harmonic_world_pointcloud_generator'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='Alessio Lovato',
    maintainer_email='alessio.lovato@iit.it',
    description='Generate and publish a synthetic world point cloud from an SDF world for Gazebo Harmonic using a ROS 2 service.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'generator_node = harmonic_world_pointcloud_generator.generator_node:main',
        ],
    },
)

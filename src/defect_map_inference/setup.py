from setuptools import setup

package_name = 'defect_map_inference'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/inference.launch.py']),
        ('share/' + package_name + '/config', ['config/inference.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Alessio Lovato',
    maintainer_email='alessio.lovato@iit.it',
    description='External Detectron2 inference service for defect mapping.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'defect_map_inference_node = defect_map_inference.inference_node:main',
        ],
    },
)

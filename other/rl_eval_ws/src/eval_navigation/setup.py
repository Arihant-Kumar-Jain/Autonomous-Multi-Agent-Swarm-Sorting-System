from setuptools import setup
import os
from glob import glob

package_name = 'eval_navigation'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='aman',
    maintainer_email='aman07112006@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bfs_navigator = eval_navigation.bfs_navigator:main',
            'q_navigator = eval_navigation.q_navigator:main',
            'dqn_navigator = eval_navigation.dqn_navigator:main',
            'metrics_node = eval_navigation.metrics_node:main',
        ],
    },
)

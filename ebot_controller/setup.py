from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ebot_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name], # This line is corrected to find the package
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools', 'numpy', 'opencv-python'],
    zip_safe=True,
    maintainer='bhaveshk01',
    maintainer_email='bhaveshk01@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    # Inside the setup() function in setup.py
    entry_points={
        'console_scripts': [
            'ebot_nav_task1A = ebot_controller.ebot_nav_task1A:main',
        ],
    },
)

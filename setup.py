from setuptools import setup, find_packages

setup(
    name='baweb_ws_sdk',
    version='0.1.1',
    packages=find_packages(),
    install_requires=[
        'websockets',
        'python-dotenv',
        # add other dependencies as needed
    ],
)

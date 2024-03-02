from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="AIPyS",
    version="0.1",
    packages=find_packages(),
    install_requires=required,
    entry_points={
        'console_scripts': [
           'aipys=AIPyS.CLI.aipys:run',  
           'updateParameters=AIPyS.CLI.set_parameters:update_user_parameters',
            'load-parameters=AIPyS.CLI.loadParameters:main', 
            ],
    },
    author="Gil Kanfer",
    author_email="gil.kanfer.il@gmail.com",
    description="AI Powered Photoswitchable Screen",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="",  # Consider adding a URL if you have a project page or source repository
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',  # Specify the minimum version of Python required
)
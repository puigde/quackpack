from setuptools import setup, find_packages

setup(
    name="quackpack",
    version="0.1.0",
    author="Your Name",
    description="Server utils",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/puigde/quackpack",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        line.strip() for line in open("requirements.txt") if line.strip()
    ],
)

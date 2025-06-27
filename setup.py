from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Dependencies are now managed in pyproject.toml
requirements = [
    "yt-dlp>=2024.1.1",
    "httpx>=0.25.0",
    "pydantic>=2.0.0",
    "click>=8.1.0",
    "PyYAML>=6.0",
    "tenacity>=8.2.0",
    "tqdm>=4.66.0",
]

setup(
    name="karaoke-video-collector",
    version="2.1.0",
    author="Karaoke Collector Team",
    description="Comprehensive tool for collecting karaoke video data from YouTube",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "karaoke-collector=collector.cli:cli",
        ],
    },
)

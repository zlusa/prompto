# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from setuptools import setup, find_packages

setup(
    name="prompto",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.31.1",
        "pandas>=2.2.0",
        "python-dotenv>=1.0.1",
        "google-cloud-aiplatform>=1.42.1",
        "Pillow>=10.2.0",
        "pyperclip>=1.8.2",
        "llama-index-core>=0.10.1",
        "google-generativeai>=0.3.2",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for analyzing and optimizing prompts using Google's Gemini model",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/zlusa/prompto",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)

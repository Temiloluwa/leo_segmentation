from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="leo_segmentation", 
    version="0.0.1",
    author="Temiloluwa Adeoti",
    author_email="temmiecvml@gmail.com",
    description="Latent Embedding Optimization for Few-shot Segmentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Temiloluwa/leo-segmentation-srp-project",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=["numpy", "pandas", "tqdm", "matplotlib"]
)
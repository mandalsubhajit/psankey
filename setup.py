import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="psankey",
    version="0.1.1",
    author="Subhajit Mandal",
    author_email="mandalsubhajit@gmail.com",
    description="Package for plotting Sankey diagrams with Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mandalsubhajit/psankey",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

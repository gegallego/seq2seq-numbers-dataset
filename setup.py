import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    INSTALL_REQUIRES = [line.split('#')[0].strip() for line in fh
                        if not line.strip().startswith('#')]

setuptools.setup(
    name='seq2seq_numbers_dataset',
    version='0.0.1',
    author='Gerard Ion Gállego Olsina',
    author_email='gerard.ion.gallego@upc.edu',
    license='MIT',
    description="Extremely simple dataset, for didactic purposes.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/gegallego/seq2seq-numbers-dataset',
    keywords="dataset corpus numbers sequence-to-sequence seq2seq",
    packages=setuptools.find_packages(),
    classifiers=[
        "Environment :: Console",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3',
    install_requires=INSTALL_REQUIRES
)
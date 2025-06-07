from setuptools import setup, find_packages

setup(
    name='keyenceutils',
    version='0.1.6',
    packages=find_packages(),
    install_requires=[
        "tifffile",
        "numpy",
        "pandas",
        # Add your dependencies here
    ],
    entry_points={
        'console_scripts': [
            # Add your console scripts here
        ],
    },
    author='Sho Yagishita',
    author_email='yagishita-tky@umin.ac.jp',
    description='Common scripts for Yagishita lab',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/ylab-common-scripts',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU GENERAL PUBLIC LICENSE',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
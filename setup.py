try:
    from setuptools import setup, find_packages
except ImportError:
    raise ImportError(
        "The 'setuptools' package is not installed. Please install it by running the following command:\n\n"
        "pip install setuptools\n"
    )

def read_requirements():
    try:
        with open('requirements.txt') as f:
            return f.read().splitlines()
    except FileNotFoundError:
        raise FileNotFoundError("Error: 'requirements.txt' not found. Please ensure the file exists in the project directory.")

setup(
    name='src',
    version='0.1',
    description='A sales forecasting framework',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Group 69',
    author_email='r2015469@novaims.unl.pt',
    url='https://github.com/M4rtinsFons3caConsulting/BusinessCases',
    license='MIT',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=read_requirements()
)

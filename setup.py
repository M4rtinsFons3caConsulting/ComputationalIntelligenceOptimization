import os

try:
    from setuptools import setup, find_packages, Extension
except ImportError:
    raise ImportError("Error: 'setuptools' is required but not installed. Please install 'setuptools' to proceed.")


# Finds all instances of '.pyx' files inside of '/cifo'
def find_pyx_files(package_dir="cifo"):
    """Find all .pyx files in the package and return them as Extension modules."""
    pyx_files = []
    for root, _, files in os.walk(package_dir):
        for file in files:
            if file.endswith(".pyx"):
                module_path = os.path.splitext(os.path.relpath(os.path.join(root, file), package_dir))[0]
                module_name = module_path.replace(os.path.sep, ".")  # Convert to module format
                pyx_files.append(Extension(f"cifo.{module_name}", sources=[os.path.join(root, file)]))
    return pyx_files


# Lazy loading cython
def get_cythonize():
    try:
        from Cython.Build import cythonize
        return cythonize
    except ImportError:
        raise ImportError("Cython is required but has not installed successfully. You may have to install it manually.")


# General setup file, reads from requirements
setup(
    name='cifo',
    version='0.1',
    description='A Genetic Algorithm Framework',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='AI',
    author_email='r2015469@novaims.unl.pt',
    url='https://github.com/M4rtinsFons3caConsulting/ComputationalIntelligenceOptimization',
    license='MIT',
    packages=find_packages(where='cifo'),
    package_dir={'': 'cifo'},
    install_requires=[
        'Cython'
    ],
    ext_modules=get_cythonize()(find_pyx_files("cifo")), # this calls the lazy loading of cythonize
)

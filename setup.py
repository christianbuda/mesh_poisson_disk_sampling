from setuptools import setup, find_packages

setup(
    name='mesh_poisson_disk_sampling',
    version='1.0',
    url='https://github.com/christianbuda/mesh_poisson_disk_sampling',
    author='Christian Buda',
    author_email='chrichri975@gmail.com',
    packages=find_packages(),
    install_requires = [
        'numpy',
        'tqdm',
        'trimesh',
        'networkx',
        'pygeodesic'
    ]
)

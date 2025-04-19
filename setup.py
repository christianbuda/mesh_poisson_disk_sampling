from setuptools import setup

setup(
    name='mesh_poisson_disk_sampling',
    version='1.0',
    url='',
    author='Christian Buda',
    author_email='chrichri975@gmail.com',
    install_requires = [
        'numpy',
        'tqdm',
        'trimesh',
        'networkx',
        'pygeodesic'
    ]
)
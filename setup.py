from setuptools import setup

setup(
    name='mesh_poisson_disk_sampling',
    version='1.0.1',
    url='https://github.com/christianbuda/mesh_poisson_disk_sampling',
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

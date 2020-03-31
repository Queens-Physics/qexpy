from setuptools import setup, find_packages

setup(
    name='qexpy',
    packages=find_packages(),
    version='2.0.0',
    description='''Package to handle error analysis and data plotting aimed at undergraduate physics.''',
    author='Astral Cai, Connor Kapahi, Prof. Ryan Martin',
    author_email='ryan.martin@queensu.ca, astral.cai@queensu.ca',
    license='GNU GLP v3',
    url='https://github.com/Queens-Physics/QExPy',
    keywords=['physics', 'laboratories', 'labs', 'undergraduate', 'data analysis', 'uncertainties', 'plotting',
              'error analysis', 'error propagation', 'uncertainty propagation'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Programming Language :: Python',
    ],
    install_requires=['numpy', 'matplotlib', 'scipy', 'IPython'],
    extras_require={
        'dev': ['pylint', 'pytest'],
        'doc': ['jupyterlab', 'sphinx', 'nbsphinx', 'nbsphinx_link', 'sphinx_autodoc_typehints', 'sphinx_rtd_theme']
    }
)

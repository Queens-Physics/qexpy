from setuptools import setup, find_packages

setup(
    name='qexpy',
    packages=find_packages(),
    version='2.0.2',
    description='''Package to handle error analysis and data plotting aimed at undergraduate physics.''',
    long_description='''QExPy (Queen’s Experimental Physics) is a Python 3 package designed to facilitate data analysis in undergraduate physics laboratories. The package contains a module to easily propagate errors in uncertainty calculations, and a module that provides an intuitive interface to plot and fit data. The package is designed to be efficient, correct, and to allow for a pedagogic introduction to error analysis. The package is extensively tested in the Jupyter Notebook environment to allow high quality reports to be generated directly from a browser.''',
    author='Astral Cai, Connor Kapahi, Prof. Ryan Martin',
    author_email='astralcai@gmail.com, ryan.martin@queensu.ca',
    license='GNU GPL v3',
    url='https://qexpy.readthedocs.io/en/latest/index.html',
    project_urls={
        "Bug Tracker": 'https://github.com/Queens-Physics/qexpy/issues',
        "Documentation": 'https://qexpy.readthedocs.io/en/latest/index.html',
        "Source Code": 'https://github.com/Queens-Physics/qexpy',
    },
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

from distutils.core import setup

setup(
    name='qexpy',
    packages=['qexpy'],
    version='0.3.7',
    description='''Package to handle error analysis and data plotting aimed
    at undergraduate physics.''',
    author='Connor Kapahi and Prof. Ryan Martin',
    author_email='ryan.martin@queensu.ca',
    license='GNU GLP v3',
    url='https://github.com/Queens-Physics/QExPy',
    download_url='https://github.com/Queens-Physics/QExPy/tarball/0.3.7',
    keywords=['physics', 'laboratories', 'labs', 'undergraduate',
              'data analysis', 'uncertainties', 'plotting', 'error analysis',
              'error propagation', 'uncertainty propagation'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Programming Language :: Python',
    ],
    install_requires=['numpy','matplotlib', 'ipywidgets', 'scipy>=0.17', 'bokeh>=0.12.1'],
)

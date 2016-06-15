from distutils.core import setup
setup(
    name='qexpy',
    packages=['qexpy'],
    version='0.0.1',
    description='''Objects to handle calculations with uncertainty and
    plotting with a focus on use in physics.''',
    author='Connor Kapahi and Prof. Ryan Martin',
    author_email='connorkapahi@gmail.com',
    license='GNU GLP v3',
    url='https://github.com/Queens-Physics/qphyssy',
    download_url='https://github.com/Queens-Physics/QExPy/tarball/0.2',
    keywords=['physics', 'undergrad', 'queens', 'university', 'analysis',
                'uncertainty', 'plotting', 'error analysis'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: GNU GLP v3',
        'Programming Language :: Python',
    ],
    install_requires=['numpy', 'scipy'],
)

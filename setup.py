import setuptools


def readme():
    with open('README.md', 'r') as f:
        return f.read()


setuptools.setup(name='annotlib',
                 version='1.0.0',
                 description='The package annotlib is a library of techniques to simulate the labeling behaviour of '
                             'real annotators.',
                 long_description=readme(),
                 long_description_content_type='text/markdown',
                 classifiers=[
                     'License :: OSI Approved :: MIT License',
                     'Programming Language :: Python :: 3.5',
                     'Programming Language :: Python :: 3.6',
                     'Programming Language :: Python :: 3.7',
                     'Programming Language :: Python :: 3.8',
                     'Operating System :: OS Independent',
                 ],
                 keywords='active learning machine learning annotator labeling',
                 url='https://annotlib.readthedocs.io/en/latest/overview.html',
                 author='Marek Herde',
                 author_email='marek.herde@uni-kassel.de',
                 license='MIT',
                 packages=setuptools.find_packages(),
                 install_requires=[
                     "numpy",
                     "scipy",
                     "scikit-learn",
                     "matplotlib",
                     "pandas",
                     "seaborn",
                     "numpy_indexed",
                 ],
                 test_suite='nose.collector',
                 tests_require=['nose'],
                 include_package_data=True,
                 zip_safe=False)

from setuptools import setup, find_packages
package_data={'bumblebee':['adjuv_functions/features_functions/data/*','adjuv_functions/sequence/data/*']}

setup(
    name = 'bumblebee',
    version = '1.0.0',
    package_dir = {'':'src'},
    packages = find_packages('src'),
    package_data = package_data,
    include_package_data=True,
    install_requires = ["numpy",
                        "scipy",
                        "pandas",
                        "matplotlib",
                        "sklearn",
                        "biopython",
                        "tensorflow",
                        "Keras",
                        "umap",
                        "gensim"],
    author = 'Ivan Gomes',
    author_email = 'ivanapg96@gmail.com',
    description = 'bumblebee - implementation of word embedding models toward protein representation and classification.',
    license = 'GNU General Public License v3',
    keywords = 'Protein Classification, Word embedding, Natural Language Processing, Machine Learning, Deep Learning',
    url = 'https://github.com/ivanapg96/bumblebee',
    long_description = open('README.rst').read(),
    long_description_content_type = 'text/markdown',
    classifiers = [
        'Development Status :: 4 - Beta',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
)

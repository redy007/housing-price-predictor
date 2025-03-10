from setuptools import setup, find_packages

setup(
    name='housing-price-predictor',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy==1.21.0',
        'pandas==1.3.0',
        'scikit-learn==0.24.2',
        'flask==2.0.1',
        'joblib==1.0.1',
        'pytest==6.2.5',
        'flake8==3.9.2'
    ]
)

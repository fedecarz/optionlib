from setuptools import setup, find_packages

setup(
    name="optionlib",
    version="0.1.0",
    description="A Python library for pricing vanilla and exotic options using Black-Scholes, Monte Carlo, Binomial and Trinomial trees.",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=1.5.0",
    ]
)
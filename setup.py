from setuptools import setup, find_packages

setup(
    name="mega_india_quant",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        # Data & Scientific Computing
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        
        # Time Series & Econometrics
        "statsmodels>=0.13.0",
        "arch>=5.0.0",
        "hmmlearn>=0.2.7",
        
        # Deep Learning
        "tensorflow>=2.8.0",
        "torch>=1.10.0",
        "pytorch-lightning>=1.6.0",
        
        # Financial
        "yfinance>=0.1.70",
        "pandas-datareader>=0.10.0",
        
        # Web Scraping & APIs
        "requests>=2.27.0",
        "beautifulsoup4>=4.10.0",
        "lxml>=4.8.0",
        
        # Utilities
        "python-dotenv>=0.19.0",
        "pyyaml>=6.0",
        "tqdm>=4.62.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        
        # Database
        "sqlalchemy>=1.4.0",
        "psycopg2-binary>=2.9.0",
        
        # Testing & Quality
        "pytest>=6.2.0",
        "black>=21.0",
        "flake8>=3.9.0",
        "mypy>=0.910",
    ],
    python_requires=">=3.8",
    author="Quant Research Team",
    description="Institutional-grade multi-module quant trading system for Indian equities",
)

# setup.py
from setuptools import setup, find_packages

setup(
    name='causal_analysis_app',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'streamlit',
        'python-dotenv',
        'openai',
    ],
    entry_points={
        'console_scripts': [
            'run_causal_app=causal_app:main',  # If you want to make it executable from the command line
        ],
    },
    author='Your Name',
    author_email='your.email@example.com',
    description='A Streamlit app for causal analysis using PyWhy-LLM.',
    long_description=open('README.md').read(),  # Create a README.md for your project
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/causal_analysis_app',  # Replace with your repository URL
    license='MIT',
)
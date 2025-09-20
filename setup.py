from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> List[str]:
    '''
    This function will return the list of requirements from the requirements.txt file
    '''
    requirements = []
    try:
        # Try reading with UTF-8 encoding first
        with open(file_path, 'r', encoding='utf-8') as file_obj:
            requirements = file_obj.readlines()
    except UnicodeDecodeError:
        # If UTF-8 fails, try UTF-16 (which might be your current encoding)
        with open(file_path, 'r', encoding='utf-16',) as file_obj:
            requirements = file_obj.readlines()
    
    requirements = [req.replace("\n", "").strip() for req in requirements if req.strip()]
    
    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)
    
    return requirements

setup(
    name='ML_Project',
    version='0.0.1',
    author='ayush prasad',
    author_email='ayush210prasad@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)

from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path: str) -> List[str]:

    unnecessary = '-e .'

    get_requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if unnecessary in requirements:
            requirements.remove(unnecessary)

    return requirements


setup(name="employee-retention",
      version='0.0.1',
      author='rprasad',
      author_email='rprasad@pccube.com', 
      packages=find_packages(),
      install_requires=get_requirements('requirements.txt')
      )
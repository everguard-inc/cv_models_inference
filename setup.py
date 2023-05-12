import os, pathlib
from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools.command.sdist import sdist
from setuptools import setup, find_packages
from subprocess import check_call

HERE = pathlib.Path(__file__).parent
PACKAGE_NAME  = 'cv_models_inference'
VERSION = '0.0.1'
AUTHOR = ''
AUTHOR_EMAIL = ''

LICENSE = 'Apache License 2.0'
DESCRIPTION = 'Package for inference cv-models'

with open("requirements.txt") as file:
    requirements = file.read().splitlines()

def gitcmd_update_submodules():
	'''	Check if the package is being deployed as a git repository. If so, recursively
		update all dependencies.

		@returns True if the package is a git repository and the modules were updated.
			False otherwise.
	'''
	if os.path.exists(os.path.join(HERE, '.git')):
		check_call(['git', 'submodule', 'update', '--init', '--recursive'])
		return True

	return False


class gitcmd_develop(develop):
	'''	Specialized packaging class that runs git submodule update --init --recursive
		as part of the update/install procedure.
	'''
	def run(self):
		gitcmd_update_submodules()
		develop.run(self)
  
  
class gitcmd_install(install):
	'''	Specialized packaging class that runs git submodule update --init --recursive
		as part of the update/install procedure.
	'''
	def run(self):
		gitcmd_update_submodules()
		install.run(self)


class gitcmd_sdist(sdist):
	'''	Specialized packaging class that runs git submodule update --init --recursive
		as part of the update/install procedure;.
	'''
	def run(self):
		gitcmd_update_submodules()
		sdist.run(self)
  
  
setup(
	name=PACKAGE_NAME,
	version=VERSION,
	description=DESCRIPTION, 
    package_dir = {"": "src"},
	install_requires=requirements,
    python_requires=">=3.8.10,<=3.8.10",
	packages = setuptools.find_packages(where="src"),
    include_package_data = True
)


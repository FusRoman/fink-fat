simple_build :
	python setup.py build
	python setup.py install

flake8:
	flake8 ../FINK-FAT

black:
	black ../FINK-FAT

clean :
	rm -r FINK_FAT.egg-info
	rm dist/FINK_FAT-0.2-py3.9.egg
	python setup.py clean --all

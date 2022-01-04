simple_build :
	python setup.py clean
	python setup.py build
	python setup.py install

flake8:
	flake8 ../Asteroids_and_Associations

black:
	black ../Asteroids_and_Associations
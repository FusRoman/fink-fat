simple_build :
	python setup.py build
	python setup.py install

flake8:
	flake8 ../Asteroids_and_Associations

black:
	black ../Asteroids_and_Associations

clean :
	rm -r Asteroids_and_Associations.egg-info
	rm dist/Asteroids_and_Associations-0.2-py3.9.egg
	python setup.py clean --all
simple_build :
<<<<<<< HEAD
=======
	python setup.py clean
>>>>>>> b06e2f212034944f3ef53702799a686f2393a179
	python setup.py build
	python setup.py install

flake8:
	flake8 ../Asteroids_and_Associations

black:
<<<<<<< HEAD
	black ../Asteroids_and_Associations

clean :
	rm -r Asteroids_and_Associations.egg-info
	rm dist/Asteroids_and_Associations-0.2-py3.9.egg
	python setup.py clean --all
=======
	black ../Asteroids_and_Associations
>>>>>>> b06e2f212034944f3ef53702799a686f2393a179

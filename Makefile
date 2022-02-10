simple_build :
	python setup.py build
	python setup.py install

flake8:
	flake8 ../fink-fat

black:
	black ../fink-fat

clean :
	rm -r fink_fat.egg-info
	rm dist/fink_fat-0.2-py3.9.egg
	rm -r dist/
	rm -r build/
	python setup.py clean --all

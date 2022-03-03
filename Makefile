simple_install :
	python -m build
	pip install .

flake8:
	flake8 ../fink-fat

black:
	black ../fink-fat

uninstall :
	pip uninstall fink_fat

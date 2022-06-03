simple_build:
	rm -rf dist/
	python -m build

to_pip_test:
	python -m twine upload --repository testpypi dist/*

upgrade:
	python -m pip install --index-url https://test.pypi.org/simple/ --no-deps fink_fat -U

flake8:
	flake8 ../fink-fat

black:
	black ../fink-fat

uninstall :
	pip uninstall fink_fat

.PHONY: prep docs test

prep:
	pip install -r requirements.txt
	pip install -e .

docs:
	cd docs && make html
	open docs/build/html/index.html

test:
	@echo 'Checking Code Styles'
	pylint qexpy
	@echo 'Running Unit Tests'
	cd tests && pytest -v --durations=0

publish:
	pip install --upgrade pip setuptools wheel
	python setup.py sdist bdist_wheel
	pip install --upgrade twine
	twine upload dist/*

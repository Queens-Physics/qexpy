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

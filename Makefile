venv: venv/bin/activate
venv/bin/activate: requirements.txt
	test -d venv || virtualenv --system-site-packages venv
	. venv/bin/activate; pip install -U -r requirements.txt
	touch venv/bin/activate
clean:
	find . -name "*.pyc" -exec rm -rf {} \;

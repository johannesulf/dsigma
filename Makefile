install_deps:
	if which pip3; then cat requirements.txt | egrep -v '#' | xargs -n 1 -L 1 pip3 install; else cat requirements.txt | egrep -v '#' | xargs -n 1 -L 1 pip install; fi

test:
	python3 tests/computeDS.py

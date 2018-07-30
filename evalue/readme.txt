[Requirements]
ubuntu:
	sudo apt install  pip
	sudo apt install python-opencv

pip:
	pip install munkres
	pip install pyyaml

[Run]
python benchmark.py -h
python benchmark.py config.yml

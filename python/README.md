

uv sync
source .venv/bin/active

python src/simulate-gt.py 41963435 1.3e-8 3.5e-9 3L-100k --ne 100000 -r 5
python simulate-gl.py 3L-100k --depths=1,5,10,20,30

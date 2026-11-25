To get setup

```
uv sync
source .venv/bin/active
cd src/simulations
```

To simulate AnoGam 3L:

```
python simulate-gt.py 41963435 1.3e-8 3.5e-9 3L-100k --ne 100000 -r 5
python simulate-gl.py 3L-100k --depths=1,5,10,20,30
python calc-kinship.py 3L-100k -p 20 -k 16 -t 516
python analyze-results 3L-100k
```

To simulate HomSap 22:

```
python simulate-gt.py 41963435 1.3e-8 3.5e-9 3L-100k --ne 100000 -r 5
python simulate-gl.py HomSap22 --depths=1,5,10,20,30
python calc-kinship.py HomSap22 -p 20 -k 16 -t 516
python analyze-results HomSap22
```

For real AnoGam 3L

```
python subsample-vcf.py gaard-3L.vcf.gz 100000 3L-GAARD-100k -r 5
python simulate-gl.py 3L-GAARD-100k --depths=1,5,10,20,30
#python calc-kinship-gl.py 3L-GAARD-100k -p 20 -k 666 -t 666
python calc-kinship-gt.py 3L-GAARD-100k -p 20 -k 100 -t 666
python calc-kinship-gl.py 3L-GAARD-100k -p 20 -k 100 -t 666
python analyze-results 3L-GAARD-100k
```

For real HomSap 22

```
wget https://storage.googleapis.com/gcp-public-data--gnomad/release/3.1.2/vcf/genomes/gnomad.genomes.v3.1.2.sites.chr22.vcf.bgz
```

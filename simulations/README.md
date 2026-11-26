To get setup

```
uv sync
source .venv/bin/active
cd src/simulations
```

## Simulate AnoGam 3L

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

## Real AnoGam

Run `download-vcf.ipynb`. Then

```
bcftools concat -Oz -o GAARD.vcf.gz GAARD-2L.vcf.gz GAARD-2R.vcf.gz GAARD-3L.vcf.gz GAARD-3R.vcf.gz
python subsample-vcf.py GAARD.vcf.gz 100000 GAARD-100k -r 1 -d 1000
python simulate-gl.py GAARD-100k --depths=1,5,10,20,30
python calc-kinship-gt.py GAARD-100k -p 20 -k 238 -t 238
python calc-kinship-gl.py GAARD-100k -p 20 -k 238 -t 238
python analyze-results 3L-GAARD-100k
```

For real HomSap 22

```
wget https://storage.googleapis.com/gcp-public-data--gnomad/release/3.1.2/vcf/genomes/gnomad.genomes.v3.1.2.sites.chr22.vcf.bgz
```

import sys

ne_name = sys.argv[1]

# have the raw VCF files + tsv
# then simulate with vcfgl to various depths
# Q30 phred score is very accurate (standard), maybe Q20 to have some error (0.01)

# in theory we could also sample the errors from a beta distribution centered at the error above with some variance (?)
# (-eq 2 --bv 0.001)

#  bcftools concat AnoGam-{2,3}{L,R}.vcf -O z -o AnoGam.vcf.gz

# we can handle all of the 4 errors no problemo (strictly convex!)
# but ngsrelate cannot. hmm, maybe use angsd to get the calls? sure?
# most people process the bam files, I think
# probably easiest just to remove the unobserved, and then stick to bi-allelic, etc.

# vcfgl can't handle multple chroms in one file
# ./vcfgl -i input.vcf.gz -o gltest -O z --depth 10 --seed 42 --error-rate 0.01
# then concat now into one big file, and plug into angsd + ngsrelate

# angsd -vcf-gl AnoGam-2L-GL.vcf.gz -domajorminor 1 -domaf 1 -doGlf 3 -out sim
# zcat sim.mafs.gz | cut -f5 | sed 1d > freqs.txt
# then only keep the first 16 samples and calculate pair-wise for that
# ngsRelate -g sim.glf.gz -n 516 -f freqs.txt -O newres -l 0.0
# then we need to manually go back to the 0-1, 2-3, etc. ugh.

# then thin (separately for each chrom), then bcftools concat the final result into one big ol file

# then run kestrel and ngsrelate on all the files, and collect the results (timing? perhaps ignore for now)

# timing will be on the human files
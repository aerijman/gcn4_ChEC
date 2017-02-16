import sys
import pandas as pd
import numpy as np

from sklearn.neighbors import KernelDensity

# 0 - Should have a list of all TSSs and their frecuency. Where frec. is > threshold (50%?), use that TSS.
# 1 - Check percentage of genes that share promoter vs number of bp of promoter from ORF (600bp - 1000bp)
# 	  some peaks are beyond 600bp from ATG... I would use 1000bp for promoter region, but I will use an optimal number
#	  after I see the results.
# 2 - Take out the genes that share peaks for the first analysis. Keep them in a separate list that will be informative later.
# 3 - list of genes with number of reads in their promoter region sorted and check if they can be clustered 
#	  into two or three bins.
# 5 - How many peaks has each gene ?
# 6 - For genes with only one peak. Where does it lay from the TSS, and for all genes, where does the median of all values lay?
# 7 - Compare number of reads from ChEC-seq to the data of MalikaSaint_et_al_2014 (mRNA or P. expression of genome upon SM induction).
# 8 - SAGA pr TFIID?
# 9 - Tables in database with NORMALIZED counts for each gene for gcn4 (+ and -SM) and meidator (+ and -SM)


# Dictionary. Will be used to access each chromosome faster
roman_numbers = {'I':'1','II':'2','III':'3','IV':'4','V':'5','VI':'6','VII':'7',\
'VIII':'8','IX':'9','X':'10','XI':'11','XII':'12','XIII':'13','XIV':'14',\
'XV':'15','XVI':'16','XVII':'17','M':'M'}


sgd_file = './SGD.tsv'
wigfile = sys.argv[1]

promoter_length = 600




from scipy import stats



def main():

	# open sgd file into a DataFrame  --> (index='locus'), columns = ['gene','chromosome','start','stop','W/C','start_promoter'] 
	global sgd
	sgd = open_sgd()

	# open wig file into a dataframe  -->  (index = index -used by w_pointer), columns = ['position','reads']
	global wig, w_pointer 
	wig, w_pointer = open_wig()
	
	counts=[]	

	# Over all genes in SGD
	for i in sgd.itertuples():

		# Check that the gene name is correct and the wig_df is not empty( so there are reads for this gene)
		if type(i.gene)==str:
			Gene_df = retrieve_positions(i.gene)
			if not Gene_df.empty:
				#print i.Index,",",i.gene,",",np.sum(Gene_df.reads.astype(int))
				#counts.append([i.Index,np.sum(Gene_df.reads.astype(int))])
				if i.gene == 'ARG3':
					print kernelPDF(Gene_df.astype(int).as_matrix())	
	#print counts


def open_sgd():
	
	# open sgd file into a dataframe
	sgd = pd.read_csv(sgd_file, delimiter='\t')

	#assign column names and use only the 'ORF' rows
	sgd.columns = ['n0','ORF_or_what','n1','locus','gene','n2','n3','n4'\
			      ,'chromosome','start','stop','W/C','n5','n6','n7','n8']
	sgd = sgd[sgd['ORF_or_what']=='ORF']

	# get rid of non-informative columns and 2-micron chromosome
	sgd = sgd.drop(['ORF_or_what','n0','n1','n2','n3','n4','n5','n6','n7','n8'],1).set_index('locus')
	sgd = sgd.ix[sgd.chromosome!='2-micron']

	# -1 to Creek and 1 to Watson
	sgd.loc[:,'W/C'][sgd['W/C']=='C']=-1
	sgd.loc[:,'W/C'][sgd['W/C']=='W']=1

	# create the column 'start promoter' and replace negative numbers \
	# with 0 (promoters that are <600bp form the start of the chromosome)
	sgd['start_promoter'] = sgd['start']-(sgd['W/C']*promoter_length)
	sgd.loc[:,'start_promoter'][sgd['start_promoter']<0]=0

	return sgd				 


def open_wig():

	# Open the wig file and make a pointer to where each chromosomes start
	wig, w_pointer = [i.strip('\n').split('\t')[:2] for i in open(wigfile)], {}
	for n in range(1,len(wig)):
		if len(wig[n])==1:
			w_pointer[roman_numbers[wig[n][0][22:]]] = n
	
	# the last pointer to the end of the file, when looking for a gene in chromosome 17 and there is no NEXT chromosome
	w_pointer['17']=n 

	# wig into dataframe
	wig = pd.DataFrame(wig)
	wig.columns = ['position','reads']

	return wig, w_pointer
	

def retrieve_positions(gene):
	
	# chromosomes as pointer for the wig file
	this_chromosome = int(sgd[sgd['gene']==gene].values[0][1])
	this = w_pointer[str(this_chromosome)]

	# assign the key to NOT to be the current chromosome
	key = [1 if this_chromosome>7 else 10][0]

	# Now go and find me the closest chromosome in the wig file (they are not in order!!)
	for i,j in w_pointer.iteritems():
		if j-this < abs(w_pointer[str(key)]-this) and j>this and i != this_chromosome:
			key = i 
	next = w_pointer[str(key)]

	# promoter positions to find in the wig 
	promoter_positions = [int(i) for i in sgd[sgd['gene']==gene][['start_promoter', 'start']].as_matrix()[0]]
	
	# In case the gene is in the Creek direction
	if sgd[sgd['gene']==gene]['W/C'].values[0]==-1:
		promoter_positions = [promoter_positions[1],promoter_positions[0]]

	# restricting the wig to the chromosome and the positions 
	Gene_df =  wig[(wig.index > this) & (wig.index < next)]

	# Only if the wig file contains counts for this gene, otherwise return False!
	if Gene_df.position.empty == False:
		Gene_df = Gene_df[(Gene_df.position.astype(int)>promoter_positions[0]) & (Gene_df.position.astype(int)<promoter_positions[1])]
		return Gene_df
	else:
		return pd.DataFrame([])


def kernelPDF(Gene_df, bandwidth=50):

	# extract x and y form the matrix Gene_df
	x,y = Gene_df[:,0], Gene_df[:,1]

	# Prepare "histogram-like" data
	histo = []
	for i in range(len(y)):
		for j in range(y[i]):
			histo.append(x[i])

	histo = np.array(histo)

	# Kernel Density Estimation with Scikit-learn
	kde = KernelDensity(kernel='epanechnikov', bandwidth=bandwidth).fit(histo[:,np.newaxis])
	pdf = np.exp(kde.score_samples(x[:,np.newaxis]))*10000

	return pdf

def kernel_SciPy(Gene_df):
	
	x,y = np.array(Gene_df['position']).astype(int), np.array(Gene_df['reads']).astype(int)

	# Prepare "histogram-like" data
	histo = []
	for i in range(len(y)):
		for j in range(y[i]):
			histo.append(x[i])

	histo = np.array(histo)

	kde = stats.gaussian_kde(x, bw_method = 0.3/y.std(ddof=1))
	
	return kde.evaluate(x)

if __name__ == '__main__':
	main()


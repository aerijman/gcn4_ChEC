# libraries for the kernel_ChEC protocol
import pandas as pd
import numpy as np
from numpy import NaN, Inf, arange, isscalar, asarray, array
import scipy as sp
import scipy.stats as st
from sklearn import mixture
from scipy.spatial.distance import cdist, pdist

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

# Dictionary to be used for translation of chromosome numbers.
roman_numbers = {'I':'1','II':'2','III':'3','IV':'4','V':'5','VI':'6','VII':'7',\
                 'VIII':'8','IX':'9','X':'10','XI':'11','XII':'12','XIII':'13','XIV':'14',\
                 'XV':'15','XVI':'16','XVII':'17','M':'M'}
inv_roman = {v: k for k, v in roman_numbers.iteritems()}


# If necessary to make an object glolbal in this libraries file
import inspect
def globalize(inputs):
	renamed_input = inputs
	callers_local_vars = inspect.currentframe().f_back.f_locals.items()
	globals()['renamed_input'] = [var_name for var_name, var_val in callers_local_vars if var_val is inputs][0]
	#print renamed_input
	#print globals()['renamed_input']


# Function used to retrieve data files from internet address 
def getFromGit(link):
    
        from html2text import html2text
        import requests    
        f = requests.get(link)
    
        return f.text.encode('ascii','replace')


# Function to open the SGD dataset
def open_sgd(promoter_length=600):

    # open sgd file into a dataframe
    global sgd
    sgd, m = getFromGit('https://raw.githubusercontent.com/aerijman/gcn4_ChEC/master/SGD.tsv'), []
    for i in sgd.split('\n'):
        m.append(i.strip().split('\t'))
    sgd = pd.DataFrame(m)    
    #sgd = pd.read_csv('./SGD.tsv', delimiter='\t')

    #assign column names and use only the 'ORF' rows
    sgd.columns = ['n0','ORF_or_what','n1','locus','gene','n2','n3','n4'\
                  ,'chromosome','start','stop','W/C','n5','n6','n7','n8']
    sgd = sgd[sgd['ORF_or_what']=='ORF']
    
    # get rid of non-informative columns and 2-micron chromosome
    sgd = sgd.drop(['ORF_or_what','n0','n1','n2','n3','n4','n5','n6','n7','n8'],1).set_index('locus')
    sgd = sgd.ix[sgd.chromosome!='2-micron']

    # -1 to Creek and 1 to Watson
    sgd.loc[sgd['W/C']=='C', 'W/C']=-1
    sgd.loc[sgd['W/C']=='W', 'W/C']=1
    
    sgd[['start','stop','W/C']] = sgd[['start','stop','W/C']].apply(pd.to_numeric) 
    
    # create the column 'start promoter' and replace negative numbers \
    # with 0 (promoters that are <600bp form the start of the chromosome)
    sgd['start_promoter'] = sgd['start']-(sgd['W/C']*promoter_length)
    sgd.loc[sgd['start_promoter']<0,'start_promoter']=0

    #sgd['promoter_median'] = (sgd['start'] + sgd['start_promoter']) / 2
    #sgd = sgd.sort_values(['chromosome','promoter_median'])
    #sgd.reset_index(inplace=True)

    return sgd


# Function to open a wig file
def open_wig(wigfile):

	# Open the wig file and make a pointer to where each chromosomes start
    #wig, w_pointer = [i.strip('\n').split('\t')[:2] for i in open(wigfile)], {} # this was for opening files, not from github  
    wig, w_pointer = wigfile, {}
    wig2, n = [], 0
    for i in wig[1:]:
        if len(i)==1:
            w_pointer[roman_numbers[i[0][22:]]] = n

        else:
            if int(i[1])>=0:                   ## era >=20...
                wig2.append(i)
                n+=1
        
    # the last pointer to the end of the file, when looking for a gene in chromosome 17 and there is no NEXT chromosome
    #w_pointer['17']=n 

    # wig into dataframe
    wig = pd.DataFrame(wig2)
    wig.columns = ['position','reads']

    return wig, w_pointer


# functino that organizes wiggle data into a dataframe
def retrieve_wiggle(locus, sgd, wig, w_pointer, region='promoter', promoter_length=600):

    # chromosomes as pointer for the wig file
    this_chromosome = int(sgd[sgd.index==locus].values[0][1])
    this = w_pointer[str(this_chromosome)]
    
    # assign the key to NOT to be the current chromosome
    key = [1 if this_chromosome>7 else 10][0]
    
    # Now go and find me the closest chromosome in the wig file (they are not in order!!)
    for i,j in w_pointer.iteritems():
        if j-this < abs(w_pointer[str(key)]-this) and j>this and i != this_chromosome:
            key = i 
    next = w_pointer[str(key)]
    
    # promoter positions to find in the wig 
    A = sgd[sgd.index==locus]['start'].values[0]
    B = (A + promoter_length*sgd.loc[sgd.index==locus,'W/C'][0]) if region=='orf' else (A - promoter_length*sgd.loc[sgd.index==locus,'W/C'][0])
    promoter_positions = [A,B] if region=='orf' else [B,A]
    
    # In case the gene is in the Creek direction
    if sgd[sgd.index==locus]['W/C'].values[0]==-1:
        promoter_positions = [promoter_positions[1],promoter_positions[0]]
    
    # restricting the wig to the chromosome and the positions 
    Gene_df =  wig[(wig.index > this) & (wig.index < next)]
    
    # Only if the wig file contains counts for this gene, otherwise return False!
    if Gene_df.position.empty == False:
        Gene_df = Gene_df[(Gene_df.position.astype(int)>promoter_positions[0]) & (Gene_df.position.astype(int)<promoter_positions[1])]
        return Gene_df.astype(int)
    else:
        return pd.DataFrame([])



# Function to obtain a pdf distribution form wiggle data
def kernelPDF(x, y, bandwidth=0, kernel_choose='epanechnikov'):
    
    #print args
    # Prepare "histogram-like" data
    histo = []
    for i in range(len(y)):
        for j in range(y[i]):
            histo.append(x[i])

    histo = np.array(histo)
    
    #busco la optima bw si no esta definida como argumento
    if locals()['bandwidth']==0:
        bandwidth = np.std(histo)*(4/(3.0*len(histo)))**(1/5)
        
        #puedo buscarla con cross-validation minimizando la MISE, que seria la mejor manera, pero tarda mucho...
        #grid = GridSearchCV(KernelDensity(kernel=kernel_choose),{'bandwidth': np.linspace(10,300,30)}, cv=9)
        #grid.fit(histo[:,None])
        #bandwidth = grid.best_params_.values()[0]
    
    #print "best_bw = ", locals()['bandwidth']
    
    # Kernel Density Estimation with Scikit-learn
    kde = KernelDensity(kernel=kernel_choose, bandwidth=bandwidth).fit(histo[:,np.newaxis])
    pdf = np.exp(kde.score_samples(x[:,np.newaxis]))
    
    return pdf, locals()['bandwidth']



# Function to get the pdf of the wiggle data normalized
def normed_PDF(gene, all_df, bandwidth=0, kernel_choose='epanechnikov'):
    
    # normalize the wiggle data
    #gene_df.loc[:,'norm_reads'] = ((gene_df.loc[:,'reads']-gene_df.reads.min())/(gene_df.reads.max()-gene_df.reads.min()))
    #orf_df.loc[:,'norm_reads'] = ((orf_df.loc[:,'reads']-orf_df.reads.min())/(orf_df.reads.max()-orf_df.reads.min()))
    #all_df.loc[:,'norm_reads'] = ((all_df.loc[:,'reads']-all_df.reads.min())/(all_df.reads.max()-all_df.reads.min()))

    # sort the data towards the pdf modeling
    all_df.sort_values(by='position', inplace=True)

    (x,y) =  zip(*[(i[0], i[1]) for i in all_df[['position','reads']].values])
    x,y = np.array(x, dtype=int), np.array(y, dtype=int)

    # best_bw = find_Kernel_best_bandwidth(x,y)
    pdf, optimal_bw = kernelPDF(x,y,bandwidth,kernel_choose='epanechnikov')

    # Normalizo y    
    y = [(i*1.0-np.min(y))/(np.max(y)-np.min(y)) for i in y]
    pdf = [(i-np.min(pdf))/(np.max(pdf)-np.min(pdf)) for i in pdf]
    # return values of x,y and the normalized pdf
    return x, y, pdf, optimal_bw


# function that retrieve statistical info about the gaussian-mixed distribution
def cuentaPeaks(sgd, wig_data, gene, promoter_length_original=600, pdf_bandwidth=0):

	df_wig, pointer_wig = wig_data#[0], wig_data[1]
	# Start by allowing the retreival of the data
	re_process, promoter_length = True, promoter_length_original

	while re_process==True and promoter_length <=2000:

		# retrieve the wiggle data from promoter, ORF and join them
		gene_df = retrieve_wiggle(sgd[sgd.gene==gene].index[0], sgd, df_wig, pointer_wig, \
			region='promoter',promoter_length=promoter_length)
		orf_df = retrieve_wiggle(sgd[sgd.gene==gene].index[0], sgd, df_wig, pointer_wig, region='orf')
		all_df = pd.DataFrame(pd.concat([gene_df,orf_df], axis=0))    
        
		# if there is no data just retrieve an empty array
		if all_df.empty:
			return []
        
		# prepare the density distribution of the wiggle data
		x, y, pdf, optimal_bw = normed_PDF(sgd[sgd.gene==gene], all_df,bandwidth=pdf_bandwidth, kernel_choose='epanechnikov')
            
		# Evaluar que la pdf empiece y termine en <0.01. Si no, actualizar el promoter_length
		# aumentando el # de nucleotidos hacia uno o los dos lados hasta que empieze y termine con menos de 0.01.
		(re_process, promoter_length) = (True,promoter_length+100) if pdf[0]>0.01 or pdf[len(pdf)-1]>0.01 \
		else (False,promoter_length)

		# reshape pdf for the fitting of the gaussian mixture to it
		pdf = np.array(pdf)[:,np.newaxis]
                    
	# for easy access to the transcription start site
	tss = sgd[sgd.gene==gene].TSS_start[0]

	# if the gene is in reverse direction, reverse the distribution data to align to other genes through the tss.
	x = [i-tss if int(sgd.loc[sgd.gene==gene, 'W/C'])==1 else tss-i for i in x]
	#plt.fill(x,pdf)

	# prepare histioram like data as input for the ML to resolve the 1D Gaussian mixtures
	histo = []
	for i in range(len(pdf)):
		for j in range(int(pdf[i]*100)):
			histo.append(x[i])
	histo = np.array(histo)[:,np.newaxis]
    
	#xp, xy_silhouette, cov, xy_elbow = np.array(x)[:,np.newaxis], [], 0, []  
	# Above was for silhouette and elbow methods now I use my own fit method that works better
	xy_fit= []
	for clusterer in range(1,10):
		g = mixture.BayesianGaussianMixture(n_components=clusterer).fit(histo)        
		classes = g.predict(np.array(x)[:,np.newaxis])[:,np.newaxis]
		#if clusterer >1:
		#    silhouette = silhouette_score(xp, classes)
		#    xy_silhouette.append((clusterer, silhouette))
     
		#cov = sum([i for i in g.covariances_])/clusterer 
		#xy_elbow.append((clusterer, cov))    
        
		# Initialize the first value of yT
		yT = st.norm.pdf(x, loc=g.means_[0][0], scale=np.sqrt(g.covariances_[0][0][0])) * g.weights_[0]
		# Then sum-up the rest of the values to model to gaussian mixed model
		for p in range(1, clusterer):
			yT = yT + st.norm.pdf(x, loc=g.means_[p][0], scale=np.sqrt(g.covariances_[p][0][0])) * g.weights_[p]
        
		yT = np.array(yT)[:,np.newaxis]
		xy_fit.append((clusterer,np.mean(cdist(yT, pdf, 'euclidean'))))
        
	return xy_fit, histo, x, y, pdf

'''
<style>
    .text_cell_render
    font-family: Times New Roman, serif
</style>
    <font color='blue'><font size="5">FUNC:calculo MISE $\int_{a}^{b} f(x) e^{2\pi i k} dx'))$ por ahora solo calculo el factor Silverman</font>

def silverman(histo):
 	n,d = len(histo), np.ndim(histo)
 	print "Just to check up... Data is taken as 1d"
 	return (n * (d + 2) / 4.)**(-1. / (d + 4))



# Use R to resolve gaussian mixtures ###

#import rpy2.interactive as r
#import rpy2.interactive.packages
#r.packages.importr("mixtools")
#rMixtools = r.packages.packages
#mixmdl = rMixtools.normalmixEM(X1)

#%load_ext rpy2.ipython
#%R library(mixtools)

#%Rpush x
%Rpush histo
%R mixmdl = normalmixEM(histo)
%R plot(mixmdl,which=2)
%R lines(density(histo, bw=1), lty=2, lwd=2)
%R mus = mixmdl
# 3 = sigma // 1 = mixing proportions (lambda) //  2 = means
%Rpull mus   

lmbda = np.array(mus[1])
means = np.array(mus[2])
sigmas = np.array(mus[3])
print('lambda = ' + str(lmbda) + ' means = ' + str(mus[2]) + 'sigmas = ' + str(mus[3]))
#for i in mus[5:6]:
#    print(i)

#silhouette_avg = silhouette_score(X1, )
'''




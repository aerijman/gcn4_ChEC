import sys
import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from ggplot import *


from numpy import NaN, Inf, arange, isscalar, asarray, array


# Dictionary. Will be used to access each chromosome faster
roman_numbers = {'I':'1','II':'2','III':'3','IV':'4','V':'5','VI':'6','VII':'7',\
'VIII':'8','IX':'9','X':'10','XI':'11','XII':'12','XIII':'13','XIV':'14',\
'XV':'15','XVI':'16','XVII':'17','M':'M'}



def open_sgd(promoter_length):
	
	# open sgd file into a dataframe
	sgd = pd.read_csv('./SGD.tsv', delimiter='\t')

	#assign column names and use only the 'ORF' rows
	sgd.columns = ['n0','ORF_or_what','n1','locus','gene','n2','n3','n4'\
			      ,'chromosome','start','stop','W/C','n5','n6','n7','n8']
	sgd = sgd[sgd['ORF_or_what']=='ORF']

	# get rid of non-informative columns and 2-micron chromosome
	sgd = sgd.drop(['ORF_or_what','n0','n1','n2','n3','n4','n5','n6','n7','n8'],1).set_index('locus')
	sgd = sgd.ix[sgd.chromosome!='2-micron']

	# -1 to Creek and 1 to Watson
	sgd[sgd['W/C']=='C'].loc[:,'W/C']=-1
	sgd.loc[sgd['W/C']=='W'][:,'W/C']=1

	# create the column 'start promoter' and replace negative numbers \
	# with 0 (promoters that are <600bp form the start of the chromosome)
	sgd['start_promoter'] = sgd['start']-(sgd['W/C']*promoter_length)
	sgd.loc[:,'start_promoter'][sgd['start_promoter']<0]=0

	# create the column 'start promoter' and replace negative numbers \
	# with 0 (promoters that are <600bp form the start of the chromosome)
	sgd['promoter_median'] = (sgd['start'] + sgd['start_promoter']) / 2
	#sgd = sgd.sort_values(['chromosome','promoter_median'])
	#sgd.reset_index(inplace=True)

	return sgd



def open_wig(wigfile):

# Open the wig file and make a pointer to where each chromosomes start
    wig, w_pointer = [i.strip('\n').split('\t')[:2] for i in open(wigfile)], {}
    wig2, n = [], 0
    for i in wig[1:]:
        if len(i)==1:
            w_pointer[roman_numbers[i[0][22:]]] = n
            print i
            #wig2.append(i)
            #n+=1
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



def retrieve_positions(locus, sgd, wig, w_pointer):

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
	promoter_positions = [int(i) for i in sgd[sgd.index==locus][['start_promoter', 'start']].as_matrix()[0]]
	
	# In case the gene is in the Creek direction
	if sgd[sgd.index==locus]['W/C'].values[0]==-1:
		promoter_positions = [promoter_positions[1],promoter_positions[0]]

	# restricting the wig to the chromosome and the positions 
	Gene_df =  wig[(wig.index > this) & (wig.index < next)]

	# Only if the wig file contains counts for this gene, otherwise return False!
	if Gene_df.position.empty == False:
		Gene_df = Gene_df[(Gene_df.position.astype(int)>promoter_positions[0]) & (Gene_df.position.astype(int)<promoter_positions[1])]
		return Gene_df
	else:
		return pd.DataFrame([])

    

def kernelPDF(x, y, bandwidth=50, kernel_choose='epanechnikov'):

	# Prepare "histogram-like" data
	histo = []
	for i in range(len(y)):
		for j in range(y[i]):
			histo.append(x[i])

	histo = np.array(histo)
    
	# Kernel Density Estimation with Scikit-learn
	kde = KernelDensity(kernel=kernel_choose, bandwidth=bandwidth).fit(histo[:,np.newaxis])
	pdf = np.exp(kde.score_samples(x[:,np.newaxis]))*10000
        
	return pdf



def ggplotIt(x,y,pdf,kernel_choose='epanechnikov'):
    print_df = pd.DataFrame([x,y,pdf]).T
    print_df.columns = ['position','wig score', kernel_choose]

    print ggplot(print_df, aes('position', 'wig score')) + geom_area() \
      + geom_line(aes('position',kernel_choose), size=2, color='red')

      
        
def peakdet(v, delta, x = None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html
    
    Returns two arrays
    
    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %      
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.
    
    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.
    
    """
    maxtab = []
    mintab = []
       
    if x is None:
        x = arange(len(v))
    
    v = asarray(v)
    
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    
    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    
    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN
    
    lookformax = True
    
    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        
        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return array(maxtab), array(mintab)

        
        
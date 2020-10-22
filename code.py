import pandas as pd
import os
import time
try:from ethnicolr import census_ln, pred_census_ln,pred_wiki_name,pred_fl_reg_name
except: os.system('pip install ethnicolr')
import seaborn as sns
import matplotlib.pylab as plt
import scipy
from itertools import permutations      
import numpy as np
import matplotlib.gridspec as gridspec
from igraph import VertexClustering
from itertools import combinations 
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.sans-serif'] = "Palatino"
plt.rcParams['font.serif'] = "Palatino"
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Palatino:italic'
plt.rcParams['mathtext.bf'] = 'Palatino:bold'
plt.rcParams['mathtext.cal'] = 'Palatino'
from matplotlib.ticker import FormatStrFormatter
from matplotlib import ticker
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor 
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.linear_model import RidgeClassifierCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA

from statsmodels.stats.multitest import multipletests

import multiprocessing
from multiprocessing import Pool 
import tqdm
import igraph
from scipy.stats import pearsonr 


global paper_df
global main_df
global g
global graphs
global pal
global homedir
global method
global node_2_a
global a_2_node
global a_2_paper
global control
global matrix_idxs
global prs
# matrix_idxs = {'white_M':0,'white_W':1,'white_U':2,'api_M':3,'api_W':4,'api_U':5,'hispanic_M':6,'hispanic_W':7,'hispanic_U':8,'black_M':9,'black_W':10,'black_U':11}


pal = np.array([[72,61,139],[82,139,139],[180,205,205],[205,129,98]])/255.

# global us_only
# us_only = True

"""
AF = author names, with the format LastName, FirstName; LastName, FirstName; etc..
SO = journal
DT = document type (review or article)
CR = reference list
TC = total citations received (at time of downloading about a year ago)
PD = month of publication
PY = year of publication
DI = DOI
"""

import argparse
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        return v

parser = argparse.ArgumentParser()
parser.add_argument('-homedir',action='store',dest='homedir',default='/Users/maxwell/Dropbox/Bertolero_Bassett_Projects/citations/')
parser.add_argument('-method',action='store',dest='method',default='wiki')
parser.add_argument('-continent',type=str2bool,action='store',dest='continent',default=False)
parser.add_argument('-continent_only',type=str2bool,action='store',dest='continent_only',default=False)
parser.add_argument('-control',type=str2bool,action='store',dest='control',default=False)
parser.add_argument('-within_poc',type=str2bool,action='store',dest='within_poc',default=False)
parser.add_argument('-walk_length',type=str,action='store',dest='walk_length',default='cited')
parser.add_argument('-walk_papers',type=str2bool,action='store',dest='walk_papers',default=False)

r = parser.parse_args()
locals().update(r.__dict__)
globals().update(r.__dict__)

wiki_2_race = {"Asian,GreaterEastAsian,EastAsian":'api', "Asian,GreaterEastAsian,Japanese":'api',
"Asian,IndianSubContinent":'api', "GreaterAfrican,Africans":'black', "GreaterAfrican,Muslim":'black',
"GreaterEuropean,British":'white', "GreaterEuropean,EastEuropean":'white',
"GreaterEuropean,Jewish":'white', "GreaterEuropean,WestEuropean,French":'white',
"GreaterEuropean,WestEuropean,Germanic":'white', "GreaterEuropean,WestEuropean,Hispanic":'hispanic',
"GreaterEuropean,WestEuropean,Italian":'white', "GreaterEuropean,WestEuropean,Nordic":'white'}

matrix_idxs = {'white_M':0,'api_M':1,'hispanic_M':2,'black_M':3,'white_W':4,'api_W':5,'hispanic_W':6,'black_W':7}

def log_p_value(p):
	if p == 0.0:
		p = "-log10($\it{p}$)>250"
	elif p > 0.001: 
		p = np.around(p,3)
		p = "$\it{p}$=%s"%(p)
	else: 
		p = (-1) * np.log10(p)
		p = "-log10($\it{p}$)=%s"%(np.around(p,0).astype(int))
	return p

def convert_r_p(r,p):
	return "$\it{r}$=%s\n%s"%(np.around(r,2),log_p_value(p))

def nan_pearsonr(x,y):
	xmask = np.isnan(x)
	ymask = np.isnan(y)
	mask = (xmask==False) & (ymask==False) 
	return pearsonr(x[mask],y[mask])

def mean_confidence_interval(data, confidence=0.95):
	a = 1.0 * np.array(data)
	n = len(a)
	m, se = np.mean(a), scipy.stats.sem(a)
	h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
	return m, m-h, m+h

def make_df(method=method):

	"""
	this makes the actual data by pulling the race from the census or wiki data
	"""
	# if os.path.exists('/%s/data/result_df_%s.csv'%(homedir,method)):
	# 	df = pd.read_csv('/%s/data/result_df_%s.csv'%(homedir,method))
	# 	return df
	main_df = pd.read_csv('/%s/article_data/NewArticleData2019_filtered.csv'%(homedir),header=0)
	result_df = pd.DataFrame(columns=['fa_race','la_race','citation_count'])
	store_fa_race = []
	store_la_race = []
	store_citations = []
	store_year = []
	store_journal = []
	store_fa_g = []
	store_la_g = []
	store_fa_category = []
	store_la_category = []
	for entry in tqdm.tqdm(main_df.iterrows(),total=len(main_df)):
		store_year.append(entry[1]['PY'])
		store_journal.append(entry[1]['SO'])
		fa = entry[1].AF.split(';')[0]
		la = entry[1].AF.split(';')[-1]
		fa_lname,fa_fname = fa.split(', ')
		la_lname,la_fname = la.split(', ')
		try:store_citations.append(len(entry[1].cited.split(',')))
		except:store_citations.append(0)
		##wiki
		if method =='wiki':
			names = [{'lname': fa_lname,'fname':fa_fname}]
			fa_df = pd.DataFrame(names,columns=['lname','fname'])
			fa_race = wiki_2_race[pred_wiki_name(fa_df,'lname','fname').race.values[0]]
			names = [{'lname': la_lname,'fname':la_fname}]
			la_df = pd.DataFrame(names,columns=['lname','fname'])
			la_race = wiki_2_race[pred_wiki_name(la_df,'lname','fname').race.values[0]]

		if method =='florida':
			names = [{'lname': fa_lname,'fname':fa_fname}]
			fa_df = pd.DataFrame(names,columns=['lname','fname'])
			fa_race = pred_fl_reg_name(fa_df,'lname','fname').race.values[0].split('_')[-1]
			names = [{'lname': la_lname,'fname':la_fname}]
			la_df = pd.DataFrame(names,columns=['lname','fname'])
			la_race = pred_fl_reg_name(la_df,'lname','fname').race.values[0].split('_')[-1]

		#census
		if method =='census':
			names = [{'name': fa_lname},{'name':la_lname}]
			la_df = pd.DataFrame(names)
			r = pred_census_ln(la_df,'name')
			fa_race,la_race= r.race.values

		
		if method =='combined':
			##wiki
			names = [{'lname': fa_lname,'fname':fa_fname}]
			fa_df = pd.DataFrame(names,columns=['fname','lname'])
			fa_race_wiki = wiki_2_race[pred_wiki_name(fa_df,'fname','lname').race.values[0]]
			names = [{'lname': la_lname,'fname':la_fname}]
			la_df = pd.DataFrame(names,columns=['fname','lname'])
			la_race_wiki = wiki_2_race[pred_wiki_name(la_df,'fname','lname').race.values[0]]


			names = [{'name': fa_lname},{'name':la_lname}]
			la_df = pd.DataFrame(names)
			r = pred_census_ln(la_df,'name')
			fa_race_census,la_race_census= r.race.values
		
			if la_race_census != la_race_wiki:
				if la_race_wiki == 'white':
					la_race = la_race_census
				if la_race_census == 'white':
					la_race = la_race_wiki
		
			elif (la_race_census != 'white') & (la_race_wiki != 'white'): la_race = la_race_wiki

			elif la_race_census == la_race_wiki: la_race = la_race_wiki


			if fa_race_census != fa_race_wiki:
				if fa_race_wiki == 'white':
					fa_race = fa_race_census
				if fa_race_census == 'white':
					fa_race = fa_race_wiki
		
			elif (fa_race_census != 'white') & (fa_race_wiki != 'white'): fa_race = fa_race_wiki

			elif fa_race_census == fa_race_wiki: fa_race = fa_race_wiki


		store_la_race.append(la_race)
		store_fa_race.append(fa_race)
		store_fa_g.append(entry[1].AG[0])
		store_la_g.append(entry[1].AG[1])
		store_fa_category.append('%s_%s' %(fa_race,entry[1].AG[0]))
		store_la_category.append('%s_%s' %(la_race,entry[1].AG[1]))
	result_df['fa_race'] = store_fa_race 
	result_df['la_race'] = store_la_race
	result_df['fa_g'] = store_fa_g
	result_df['la_g'] = store_la_g
	result_df['journal'] = store_journal
	result_df['year'] = store_year
	result_df['citation_count'] = store_citations
	result_df['fa_category'] = store_fa_category
	result_df['la_category'] = store_la_category
	# result_df.citation_count = result_df.citation_count.values.astype(int) 
	result_df.to_csv('/%s/data/result_df_%s.csv'%(homedir,method),index=False)
	return result_df

def make_pr_df(method=method):

	"""
	this makes the actual data by pulling the race from the census or wiki data
	"""
	main_df = pd.read_csv('/%s/article_data/NewArticleData2019.csv'%(homedir),header=0)
	prs = np.zeros((main_df.shape[0],8,8))

	gender_base = {}
	for year in np.unique(main_df.PY.values):
		ydf = main_df[main_df.PY==year].AG
		fa = np.array([x[0] for x in ydf.values])
		la = np.array([x[1] for x in ydf.values])

		fa_m = len(fa[fa=='M'])/ len(fa[fa!='U'])
		fa_w = len(fa[fa=='W'])/ len(fa[fa!='U'])

		la_m = len(la[fa=='M'])/ len(la[la!='U'])
		la_w = len(la[fa=='W'])/ len(la[la!='U'])

		gender_base[year] = [fa_m,fa_w,la_m,la_w]

	asian = [0,1,2]
	black = [3,4]
	white = [5,6,7,8,9,11,12]
	hispanic = [10]
	if method =='wiki_black':
		black = [3]
	for entry in tqdm.tqdm(main_df.iterrows(),total=len(main_df)):
	
		fa = entry[1].AF.split(';')[0]
		la = entry[1].AF.split(';')[-1]
		fa_lname,fa_fname = fa.split(', ')
		la_lname,la_fname = la.split(', ')
		fa_g = entry[1].AG[0]   
		la_g = entry[1].AG[1]	
		paper_matrix = np.zeros((2,8))
		# 1/0
		##wiki
		if method =='wiki' or method == 'wiki_black':
			names = [{'lname': fa_lname,'fname':fa_fname}]
			fa_df = pd.DataFrame(names,columns=['lname','fname'])
			
			fa_race = pred_wiki_name(fa_df,'lname','fname').values[0][3:]
			fa_race = [np.sum(fa_race[white]),np.sum(fa_race[asian]),np.sum(fa_race[hispanic]),np.sum(fa_race[black])]
			
			names = [{'lname': la_lname,'fname':la_fname}]
			la_df = pd.DataFrame(names,columns=['lname','fname'])
			
			la_race = pred_wiki_name(la_df,'lname','fname').values[0][3:]
			la_race = [np.sum(la_race[white]),np.sum(la_race[asian]),np.sum(la_race[hispanic]),np.sum(la_race[black])]

		# #census
		if method =='census':
			names = [{'name': fa_lname},{'name':la_lname}]
			la_df = pd.DataFrame(names)
			r = pred_census_ln(la_df,'name')
			fa_race = [r.iloc[0]['white'],r.iloc[0]['api'],r.iloc[0]['hispanic'],r.iloc[0]['black']]
			la_race = [r.iloc[1]['white'],r.iloc[1]['api'],r.iloc[1]['hispanic'],r.iloc[1]['black']]
		
		if method =='florida':

			names = [{'lname': fa_lname,'fname':fa_fname}]
			fa_df = pd.DataFrame(names,columns=['lname','fname'])
			asian, hispanic, black, white = pred_fl_reg_name(fa_df,'lname','fname').values[0][3:] 
			fa_race = [white,asian,hispanic,black]
			
			names = [{'lname': la_lname,'fname':la_fname}]
			la_df = pd.DataFrame(names,columns=['lname','fname'])
			asian, hispanic, black, white = pred_fl_reg_name(la_df,'lname','fname').values[0][3:] 
			la_race = [white,asian,hispanic,black]



		if method == 'combined':
			names = [{'lname': fa_lname,'fname':fa_fname}]
			fa_df = pd.DataFrame(names,columns=['fname','lname'])
			fa_race_wiki = pred_wiki_name(fa_df,'lname','fname').values[0][3:]
			fa_race_wiki  = [np.sum(fa_race_wiki[white]),np.sum(fa_race_wiki[asian]),np.sum(fa_race_wiki[hispanic]),np.sum(fa_race_wiki[black])]
			
			names = [{'lname': la_lname,'fname':la_fname}]
			la_df = pd.DataFrame(names,columns=['fname','lname'])
			la_race_wiki = pred_wiki_name(la_df,'lname','fname').values[0][3:]
			la_race_wiki = [np.sum(la_race_wiki[white]),np.sum(la_race_wiki[asian]),np.sum(la_race_wiki[hispanic]),np.sum(la_race_wiki[black])]


			names = [{'name': fa_lname},{'name':la_lname}]
			la_df = pd.DataFrame(names)
			r = pred_census_ln(la_df,'name')
			fa_race_census = [r.iloc[0]['white'],r.iloc[0]['api'],r.iloc[0]['hispanic'],r.iloc[0]['black']]
			la_race_census = [r.iloc[1]['white'],r.iloc[1]['api'],r.iloc[1]['hispanic'],r.iloc[1]['black']]
			
			if fa_race_census[0] < fa_race_wiki[0]: fa_race = fa_race_census
			else: fa_race = fa_race_wiki
			if la_race_census[0] < la_race_wiki[0]: la_race = la_race_census
			else: la_race = la_race_wiki


		gender_b = gender_base[year]
		if fa_g == 'M': paper_matrix[0] = np.outer([1,0],fa_race).flatten() 
		if fa_g == 'W': paper_matrix[0] = np.outer([0,1],fa_race).flatten() 
		if fa_g == 'U': paper_matrix[0] = np.outer([gender_b[0],gender_b[1]],fa_race).flatten() 

		if la_g == 'M': paper_matrix[1] = np.outer([1,0],la_race).flatten() 
		if la_g == 'W': paper_matrix[1] = np.outer([0,1],la_race).flatten() 
		if la_g == 'U': paper_matrix[1] = np.outer([gender_b[2],gender_b[3]],la_race).flatten() 

		paper_matrix = np.outer(paper_matrix[0],paper_matrix[1]) 
		paper_matrix = paper_matrix / np.sum(paper_matrix)
		prs[entry[0]] = paper_matrix
	
	np.save('/%s/data/result_pr_df_%s.npy'%(homedir,method),prs)

def make_all_author_race():

	"""
	this makes the actual data by pulling the race from the census or wiki data,
	but this version include middle authors, which we use for the co-authorship networks
	"""
	main_df = pd.read_csv('/%s/article_data/NewArticleData2019.csv'%(homedir),header=0)
	names = []
	lnames = []
	fnames = []
	for entry in main_df.iterrows():
		for a in entry[1].AF.split('; '):
			a_lname,a_fname = a.split(', ')
			lnames.append(a_lname.strip())
			fnames.append(a_fname.strip())
			names.append(a)

	df = pd.DataFrame(np.array([names,fnames,lnames]).swapaxes(0,1),columns=['name','fname','lname'])
	df = df.drop_duplicates('name') 

	if method =='florida':
		# 1/0
		r = pred_fl_reg_name(df,'lname','fname')
		r.rename(columns={'nh_black':'black','nh_white':'white'})  
		r.to_csv('/%s/data/result_df_%s_all.csv'%(homedir,method),index=False)


	if method =='census':
		r = pred_census_ln(df,'lname')
		r.to_csv('/%s/data/result_df_%s_all.csv'%(homedir,method),index=False)
	
		all_races = []
		r = dict(zip(df.name.values,df.race.values)) 
		for idx,paper in tqdm.tqdm(main_df.iterrows(),total=main_df.shape[0]):
			races = []
			for a in paper.AF.split('; '):
				a_lname,a_fname = a.split(', ')
				races.append(r[a_lname.strip()])
			all_races.append('_'.join(str(x) for x in races))
		main_df['all_races'] = all_races 
		main_df.to_csv('/%s/data/all_data_%s.csv'%(homedir,method),index=False)


	race2wiki = {'api': ["Asian,GreaterEastAsian,EastAsian","Asian,GreaterEastAsian,Japanese", "Asian,IndianSubContinent"],
	'black':["GreaterAfrican,Africans", "GreaterAfrican,Muslim"],
	'white':["GreaterEuropean,British", "GreaterEuropean,EastEuropean", "GreaterEuropean,Jewish", "GreaterEuropean,WestEuropean,French",
	"GreaterEuropean,WestEuropean,Germanic", "GreaterEuropean,WestEuropean,Nordic", "GreaterEuropean,WestEuropean,Italian"],
	'hispanic':["GreaterEuropean,WestEuropean,Hispanic"]}

	if method =='wiki':
		r = pred_wiki_name(df,'lname','fname')
		for race in ['api','black','hispanic','white']:
			r[race] = 0.0
			for e in race2wiki[race]:
				r[race] = r[race] + r[e]
				
		for race in ['api','black','hispanic','white']:
			for e in race2wiki[race]:	
				r  = r.drop(columns=[e])
		r.to_csv('/%s/data/result_df_%s_all.csv'%(homedir,method),index=False)	
		

		all_races = []
		for idx,paper in tqdm.tqdm(main_df.iterrows(),total=main_df.shape[0]):
			races = []
			for a in paper.AF.split('; '):
				races.append(r[r.name==a].race.values[0])
			all_races.append('_'.join(str(x) for x in races))
		main_df['all_races'] = all_races
		main_df.to_csv('/%s/data/all_data_%s.csv'%(homedir,method),index=False)

	if method =='combined':
		r_wiki = pred_wiki_name(df,'lname','fname')
		
		for race in ['api','black','hispanic','white']:
			r_wiki[race] = 0.0
			for e in race2wiki[race]:
				r_wiki[race] = r_wiki[race] + r_wiki[e]
				
		for race in ['api','black','hispanic','white']:
			for e in race2wiki[race]:	
				r_wiki  = r_wiki.drop(columns=[e])

		r_census = pred_census_ln(df,'lname')

		census = r_census.white < r_wiki.white 
		wiki = r_census.white > r_wiki.white 

		r = r_census.copy()
		r[census] = r_census
		r[wiki] = r_wiki
		r.to_csv('/%s/data/result_df_%s_all.csv'%(homedir,method),index=False)	

def figure_1_pr_authors():

	df = pd.read_csv('/%s/data/result_df_%s_all.csv'%(homedir,method))
	paper_df = pd.read_csv('/%s/article_data/NewArticleData2019.csv'%(homedir),header=0)

	results = []
	for year in np.unique(paper_df.PY.values):
		print (year)
		ydf = paper_df[paper_df.PY==year]
		names = []
		for p in ydf.iterrows():
			for n in p[1].AF.split(';'):
				names.append(n.strip())
		names = np.unique(names)
		result = np.zeros((len(names),4))
		for idx,name in enumerate(names):
			try:result[idx] = df[df.name==name].values[0][-4:] 
			except:result[idx] = np.nan
		results.append(np.nansum(result,axis=0))
	results = np.array(results)
	plt.close()
	sns.set(style='white',font='Palatino')
	# pal = sns.color_palette("Set2")
	# pal = sns.color_palette("vlag",4)
	fig = plt.figure(figsize=(7.5,4),constrained_layout=False)
	gs = gridspec.GridSpec(15, 14, figure=fig,wspace=.75,hspace=0,left=.1,right=.9,top=.9,bottom=.1)
	ax1 = fig.add_subplot(gs[:15,:7])

	ax1_plot = plt.stackplot(np.unique(paper_df.PY),np.flip(results.transpose()[[3,0,2,1]],axis=0), labels=['Black','Hispanic','Asian','White'],colors=np.flip(pal,axis=0), alpha=1)
	handles, labels = plt.gca().get_legend_handles_labels()
	labels.reverse()
	handles.reverse()
	leg = plt.legend(loc=2,frameon=False,labels=labels,handles=handles,fontsize=8)
	for text in leg.get_texts():
		plt.setp(text, color = 'black')
	plt.margins(0,0)
	plt.ylabel('sum of predicted author race')
	plt.xlabel('publication year')
	
	ax1.tick_params(axis='y', which='major', pad=0)
	plt.title('a',{'fontweight':'bold'},'left',pad=2)
	# 1/0


	ax2 = fig.add_subplot(gs[:15,8:])

	ax2_plot = plt.stackplot(np.unique(paper_df.PY),np.flip(np.divide(results.transpose()[[3,0,2,1]],np.sum(results,axis=1)),axis=0)*100, labels=['Black','Hispanic','Asian','White'],colors=np.flip(pal,axis=0),alpha=1)
	handles, labels = plt.gca().get_legend_handles_labels()
	labels.reverse()
	handles.reverse()
	leg = plt.legend(loc=2,frameon=False,labels=labels,handles=handles,fontsize=8)
	for text in leg.get_texts():
		plt.setp(text, color = 'white')
	plt.margins(0,0)
	plt.ylabel('percentage of predicted author race',labelpad=-5)
	plt.xlabel('publication year')
	ax2.yaxis.set_major_formatter(ticker.PercentFormatter()) 
	ax2.tick_params(axis='y', which='major', pad=0)
	plt.title('b',{'fontweight':'bold'},'left',pad=2)
	plt.savefig('authors.pdf')

def figure_1_pr():
	n_iters = 1000
	df =pd.read_csv('/%s/article_data/NewArticleData2019.csv'%(homedir),header=0).rename({'PY':'year','SO':'journal'},axis='columns')  
	matrix = np.load('/%s/data/result_pr_df_%s.npy'%(homedir,method))
	results = np.zeros((len(np.unique(df.year)),4))

	if within_poc == False:
		labels = ['white author & white author','white author & author of color','author of color & white author','author of color &\nauthor of color']
		groups = [np.vectorize(matrix_idxs.get)(['white_M','white_W',]),
		np.vectorize(matrix_idxs.get)(['api_M','api_W','hispanic_M','hispanic_W','black_M','black_W',])]
		names = ['white-white','white-poc','poc-white','poc-poc']

		plot_matrix = np.zeros((matrix.shape[0],len(groups),len(groups)))
		plot_base_matrix = np.zeros((matrix.shape[0],len(groups),len(groups)))

		for i in range(len(groups)):
			for j in range(len(groups)):
				plot_matrix[:,i,j] = np.nansum(matrix[:,groups[i]][:,:,groups[j]].reshape(matrix.shape[0],-1),axis=1)

		for yidx,year in enumerate(np.unique(df.year)):
			papers = df[df.year==year].index  
			r = np.mean(plot_matrix[papers],axis=0).flatten()
			results[yidx,0] = r[0]
			results[yidx,1] = r[1]
			results[yidx,2] = r[2]
			results[yidx,3] = r[3]

	if within_poc == True:
		names = ['white author','Asian author','Hispanic author','Black author']
		groups = [[0,4],[1,5],[2,6],[3,7]]
		labels = names

		plot_matrix = np.zeros((matrix.shape[0],len(groups)))

		for i in range(4):
			plot_matrix[:,i] = plot_matrix[:,i] + np.nansum(np.nanmean(matrix[:,groups[i],:],axis=-1),axis=-1)  
			plot_matrix[:,i] = plot_matrix[:,i] + np.nansum(np.nanmean(matrix[:,:,groups[i]],axis=-1),axis=-1)  

		for yidx,year in enumerate(np.unique(df.year)):
			papers = df[df.year==year].index  
			r = np.mean(plot_matrix[papers],axis=0).flatten()
			results[yidx,0] = r[0]
			results[yidx,1] = r[1]
			results[yidx,2] = r[2]
			results[yidx,3] = r[3]
	
	plt.close()
	sns.set(style='white',font='Palatino')
	# pal = sns.color_palette("Set2")
	# pal = sns.color_palette("vlag",4)
	fig = plt.figure(figsize=(7.5,4),constrained_layout=False)
	gs = gridspec.GridSpec(15, 16, figure=fig,wspace=.75,hspace=0,left=.1,right=.9,top=.9,bottom=.1)

	ax1 = fig.add_subplot(gs[:15,:5])
	plt.sca(ax1)
	ax1_plot = plt.stackplot(np.unique(df.year),np.flip(results.transpose(),axis=0)*100, labels=np.flip(labels),colors=np.flip(pal,axis=0), alpha=1)
	handles, labels = plt.gca().get_legend_handles_labels()
	labels.reverse()
	handles.reverse()
	leg = plt.legend(loc=9,frameon=False,labels=labels,handles=handles,fontsize=8)
	for text in leg.get_texts():
		plt.setp(text, color = 'w')
	plt.margins(0,0)
	plt.ylabel('percentage of publications')
	plt.xlabel('publication year')
	ax1.tick_params(axis='x', which='major', pad=-1)
	ax1.tick_params(axis='y', which='major', pad=0)
	i,j,k,l = np.flip(results[0]*100)
	i,j,k,l = [i,(i+j),(i+j+k),(i+j+k+l)]
	i,j,k,l = [np.mean([0,i]),np.mean([i,j]),np.mean([j,k]),np.mean([k,l])]
	# i,j,k,l = np.array([100]) - np.array([i,j,k,l])  
	plt.sca(ax1)	
	ax1.yaxis.set_major_formatter(ticker.PercentFormatter()) 
	ax1.set_yticks([i,j,k,l])
	ax1.set_yticklabels(np.flip(np.around(results[0]*100,0).astype(int)))

	ax2 = ax1_plot[0].axes.twinx()
	plt.sca(ax2)
	i,j,k,l = np.flip(results[-1]*100)
	i,j,k,l = [i,(i+j),(i+j+k),(i+j+k+l)]
	i,j,k,l = [np.mean([0,i]),np.mean([i,j]),np.mean([j,k]),np.mean([k,l])] 
	plt.ylim(0,100)
	ax2.yaxis.set_major_formatter(ticker.PercentFormatter()) 
	ax2.set_yticks([i,j,k,l])
	ax2.set_yticklabels(np.flip(np.around(results[-1]*100,0)).astype(int))
	plt.xticks([1995., 2000., 2005., 2010., 2015., 2019],np.array([1995., 2000., 2005., 2010., 2015., 2019]).astype(int))   

	ax2.tick_params(axis='y', which='major', pad=0)
	plt.title('a',{'fontweight':'bold'},'left',pad=2)

	plot_df = pd.DataFrame(columns=['year','percentage','iteration'])  
	for yidx,year in enumerate(np.unique(df.year)):
		for i in range(n_iters):
			data = df[(df.year==year)]
			papers = data.sample(int(len(data)),replace=True).index
			r = np.mean(plot_matrix[papers],axis=0).flatten()
			total = r.sum()
			r = np.array(r[1:])/total
			r = r.sum()
			tdf = pd.DataFrame(np.array([r,year,i]).reshape(1,-1),columns=['percentage','year','iteration']) 
			plot_df = plot_df.append(tdf,ignore_index=True)


	plot_df.percentage = plot_df.percentage.astype(float)
	plot_df.iteration = plot_df.iteration.astype(int)
	plot_df.percentage = plot_df.percentage.astype(float) * 100
	pct_df = pd.DataFrame(columns=['year','percentage','iteration'])
	plot_df = plot_df.sort_values('year')
	for i in range(n_iters):
		a = plot_df[(plot_df.iteration==i)].percentage.values
		# change = np.diff(a) / a[:-1] * 100.
		change = np.diff(a) 
		tdf = pd.DataFrame(columns=['year','percentage','iteration'])
		tdf.year = range(1997,2020)
		tdf.percentage = change[1:]
		tdf.iteration = i
		pct_df = pct_df.append(tdf,ignore_index=True)

	pct_df = pct_df.dropna()
	pct_df = pct_df[np.isinf(pct_df.percentage)==False] 
	ci = mean_confidence_interval(pct_df.percentage)
	ci = np.around(ci,2)
	print ("Across 1000 bootstraps, the mean percent increase per year was %s%% (95 CI:%s%%,%s%%)"%(ci[0],ci[1],ci[2]))

	plt.text(.5,.48,"Increasing at %s%% per year\n(95%% CI:%s%%,%s%%)"%(ci[0],ci[1],ci[2]),{'fontsize':8,'color':'white'},horizontalalignment='center',verticalalignment='bottom',rotation=9,transform=ax2.transAxes)

	axes = []
	jidx = 3
	for makea in range(5):
		axes.append(fig.add_subplot(gs[jidx-3:jidx,6:10]))
		jidx=jidx+3
	
	for aidx,journal in enumerate(np.unique(df.journal)):
		ax = axes[aidx]
		plt.sca(ax)
		if aidx == 2: ax.set_ylabel('percentage of publications')
		if aidx == 4: ax.set_xlabel('publication\nyear',labelpad=-10)
		results = np.zeros(( len(np.unique(df[(df.journal==journal)].year)),4))
		for yidx,year in enumerate(np.unique(df[(df.journal==journal)].year)):
			papers = df[(df.year==year)&(df.journal==journal)].index
			r = np.mean(plot_matrix[papers],axis=0).flatten()
			results[yidx,0] = r[0]
			results[yidx,1] = r[1]
			results[yidx,2] = r[2]
			results[yidx,3] = r[3]

		data = df[df.journal==journal]
		if journal == 'NATURE NEUROSCIENCE':
			for i in range(3): results = np.concatenate([[[0,0,0,0]],results],axis=0) 
		ax1_plot = plt.stackplot(np.unique(df.year),np.flip(results.transpose(),axis=0)*100, labels=np.flip(labels,axis=0),colors=np.flip(pal,axis=0), alpha=1)
		plt.margins(0,0)
		ax.set_yticks([])
		if aidx != 4:
			ax.set_xticks([])
		else: plt.xticks(np.array([1996.5,2017.5]),np.array([1995.,2019]).astype(int)) 
		plt.title(journal.title(), pad=-10,color='w',fontsize=8)
		if aidx == 0: plt.text(0,1,'b',{'fontweight':'bold'},horizontalalignment='left',verticalalignment='bottom',transform=ax.transAxes)


	journals = np.unique(df.journal)
	plot_df = pd.DataFrame(columns=['journal','year','percentage','iteration'])  
	for j in journals:
		for yidx,year in enumerate(np.unique(df.year)):
			for i in range(n_iters):
				data = df[(df.year==year)&(df.journal==j)]
				papers = data.sample(int(len(data)),replace=True).index
				r = np.mean(plot_matrix[papers],axis=0).flatten()
				total = r.sum()
				r = np.array(r[1:])/total
				r = r.sum()
				tdf = pd.DataFrame(np.array([j,r,year,i]).reshape(1,-1),columns=['journal','percentage','year','iteration']) 
				plot_df = plot_df.append(tdf,ignore_index=True)

	plot_df.percentage = plot_df.percentage.astype(float)
	plot_df.iteration = plot_df.iteration.astype(int)
	plot_df.percentage = plot_df.percentage.astype(float) * 100
	pct_df = pd.DataFrame(columns=['journal','year','percentage','iteration'])
	plot_df = plot_df.sort_values('year')
	for i in range(n_iters):
		for j in journals:
			a = plot_df[(plot_df.iteration==i)&(plot_df.journal==j)].percentage.values
			# change = np.diff(a) / a[:-1] * 100.
			change = np.diff(a)
			tdf = pd.DataFrame(columns=['journal','year','percentage','iteration'])
			tdf.year = range(1997,2020)
			tdf.percentage = change[1:]
			tdf.journal = j
			tdf.iteration = i
			pct_df = pct_df.append(tdf,ignore_index=True)
	
	pct_df = pct_df.dropna()
	pct_df = pct_df[np.isinf(pct_df.percentage)==False] 
	ci = pct_df.groupby(['journal']).percentage.agg(mean_confidence_interval).values 

	axes = []
	jidx = 3
	for makea in range(5):
		axes.append(fig.add_subplot(gs[jidx-3:jidx,11:]))
		jidx=jidx+3 

	for i,ax,journal,color in zip(range(5),axes,journals,sns.color_palette("rocket_r", 5)):
		plt.sca(ax)
		ax.clear()
		# 
		# plot_df[np.isnan(plot_df.percentage)] = 0.0
		if i == 0: plt.text(0,1,'c',{'fontweight':'bold'},horizontalalignment='left',verticalalignment='bottom',transform=ax.transAxes)
		lp = sns.lineplot(data=plot_df[plot_df.journal==journal],y='percentage',x='year',color=color,ci='sd')   
		plt.margins(0,0)

		thisdf = plot_df[plot_df.journal==journal]
		minp = int(np.around(thisdf.mean()['percentage'],0))
		thisdf = thisdf[thisdf.year==thisdf.year.max()]
		maxp = int(np.around(thisdf.mean()['percentage'],0))
		
		plt.text(-0.01,.5,'%s'%(minp),horizontalalignment='right',verticalalignment='top', transform=ax.transAxes,fontsize=10)
		plt.text(1.01,.9,'%s'%(maxp),horizontalalignment='left',verticalalignment='top', transform=ax.transAxes,fontsize=10)
		ax.set_yticks([])
		# ax.set_xticks([])
		ax.set_ylabel('')
		plt.margins(0,0)
		ax.set_yticks([])
		if i == 2:
			ax.set_ylabel('percentage of publications',labelpad=12)
		if i != 4: ax.set_xticks([])

		
		else: plt.xticks(np.array([1.5,22.5]),np.array([1995.,2019]).astype(int)) 
		mean_pc,min_pc,max_pc = np.around(ci[i],2)
		if i == 4: ax.set_xlabel('publication\nyear',labelpad=-10)
		else: ax.set_xlabel('')
		plt.text(.99,0,'95%' + "CI: %s<%s<%s"%(min_pc,mean_pc,max_pc),horizontalalignment='right',verticalalignment='bottom', transform=ax.transAxes,fontsize=8)
		if journal == 'NATURE NEUROSCIENCE':
			plt.xlim(-3,21)
	plt.savefig('/%s/figures/figure1_pr_%s_%s.pdf'%(homedir,method,within_poc))

def validate():
	black_names = pd.read_csv('%s/data/Black scientists - Faculty.csv'%(homedir))['Name'].values[1:]
	fnames = []
	lnames = []
	all_names =[]
	for n in black_names:
		try:
			fn,la = n.split(' ')[:2]
			fnames.append(fn.strip())
			lnames.append(la.strip())
			all_names.append('%s_%s'%(fn.strip(),la.strip()))
		except:continue

	black_df = pd.DataFrame(np.array([all_names,fnames,lnames]).swapaxes(0,1),columns=['name','fname','lname'])

	main_df = pd.read_csv('/%s/article_data/NewArticleData2019.csv'%(homedir),header=0)
	names = []
	lnames = []
	fnames = []
	for entry in main_df.iterrows():
		for a in entry[1].AF.split('; '):
			a_lname,a_fname = a.split(', ')
			lnames.append(a_lname.strip())
			fnames.append(a_fname.strip())
			names.append('%s_%s'%(a_fname,a_lname))
	main_df = pd.DataFrame(np.array([names,fnames,lnames]).swapaxes(0,1),columns=['name','fname','lname'])
	main_df = main_df.drop_duplicates('name') 





	if method == 'wiki':
		black_r = pred_wiki_name(black_df,'lname','fname')
		all_r = pred_wiki_name(main_df,'lname','fname')
		asian = [0,1,2]
		black = [3,4]
		white = [5,6,7,8,9,11,12]
		hispanic = [10]

		all_df = pd.DataFrame(columns=['probability','sample'])
		all_df['probability'] = all_r.as_matrix()[:,4:][:,black].sum(axis=1)
		all_df['sample'] = 'papers'
		black_df = pd.DataFrame(columns=['probability','sample'])
		black_df['probability'] = black_r.as_matrix()[:,4:][:,black].sum(axis=1)
		black_df['sample'] = 'Black-in-STEM'

	if method == 'florida':
		black_r = pred_fl_reg_name(black_df,'lname','fname')
		all_r = pred_fl_reg_name(main_df,'lname','fname')
		asian = [0,1,2]
		black = [3,4]
		white = [5,6,7,8,9,11,12]
		hispanic = [10]

		all_df = pd.DataFrame(columns=['probability','sample'])
		all_df['probability'] = all_r.values[:,-2]
		all_df['sample'] = 'papers'
		black_df = pd.DataFrame(columns=['probability','sample'])
		black_df['probability'] = black_r.values[:,-2]  
		black_df['sample'] = 'Black-in-STEM'

		 
	if method == 'census':
		black_r = pred_census_ln(black_df,'lname')
		all_r = pred_census_ln(main_df,'lname')

		all_df = pd.DataFrame(columns=['probability','sample'])
		all_df['probability'] = all_r.values[:,-3]
		all_df['sample'] = 'papers'
		black_df = pd.DataFrame(columns=['probability','sample'])
		black_df['probability'] = black_r.values[:,-3]   
		black_df['sample'] = 'Black-in-STEM'


	data = all_df.append(black_df,ignore_index=True)
	data.probability = data.probability.astype(float)  
	
	plt.close()
	sns.set(style='white',font='Palatino')
	fig = plt.figure(figsize=(7.5,3),constrained_layout=True)
	gs = gridspec.GridSpec(6,6, figure=fig)

	ax1 = fig.add_subplot(gs[:6,:3])
	plt.sca(ax1)	

	sns.histplot(data=data,x='probability',hue="sample",stat='density',common_norm=False,bins=20)  
	ax2 = fig.add_subplot(gs[:6,3:])
	plt.sca(ax2)
	sns.histplot(data=data,x='probability',hue="sample",stat='density',common_norm=False,bins=20)  
	plt.ylim(0,2.5)
	plt.savefig('Black-in-STEM_%s.pdf'%(method))


	plt.close()
	sns.set(style='white',font='Palatino')
	fig = plt.figure(figsize=(7.5,3),constrained_layout=True)
	gs = gridspec.GridSpec(6,6, figure=fig)
	ax1 = fig.add_subplot(gs[:6,:3])
	plt.sca(ax1)	

	sns.histplot(data=data[data['sample']=='papers'],x='probability',stat='density',common_norm=False,bins=20)  
	ax2 = fig.add_subplot(gs[:6,3:])
	plt.sca(ax2)
	sns.histplot(data=data[data['sample']=='Black-in-STEM'],x='probability',hue="sample",stat='density',common_norm=False,bins=20)  
	# plt.ylim(0,2.5)
	plt.savefig('Black-in-STEM_2.pdf')	

def make_pr_control():
	"""
	control for features of citing article
	"""
	# 1) the year of publication
	# 2) the journal in which it was published
	# 3) the number of authors
	# 4) whether the paper was a review article
	# 5) the seniority of the paper’s first and last authors.
	# 6) paper location
	df = pd.read_csv('/%s/article_data/NewArticleData2019.csv'%(homedir),header=0)
	prs = np.load('/%s/data/result_pr_df_%s.npy'%(homedir,method))
	cont = pd.read_csv('/%s/article_data/CountryAndContData.csv'%(homedir)) 
	df = df.merge(cont,how='outer',left_index=True, right_index=True)
	df = df.merge(pd.read_csv('/%s/article_data/SeniorityData.csv'%(homedir)),left_index=True, right_index=True)

	reg_df = pd.DataFrame(columns=['year','n_authors','journal','paper_type','senior','location'])
	
	for entry in tqdm.tqdm(df.iterrows(),total=len(df)):
		idx = entry[0]
		paper = entry[1]
		year = entry[1].PY
		n_authors = len(paper.AF.split(';'))
		journal = entry[1].SO
		paper_type = paper.DT
		senior = entry[1].V4
		try: loc = entry[1]['FirstListed.Cont'].split()[0]
		except: loc = 'None'
		reg_df.loc[len(reg_df)] = [year,n_authors,journal,paper_type,senior,loc]

	reg_df["n_authors"] = pd.to_numeric(reg_df["n_authors"])
	reg_df["year"] = pd.to_numeric(reg_df["year"])
	reg_df["senior"] = pd.to_numeric(reg_df["senior"])
	
	skl_df = pd.get_dummies(reg_df).values

	ridge = MultiOutputRegressor(RidgeCV(alphas=[1e-5,1e-4,1e-3, 1e-2, 1e-1, 1,10,25,50,75,100])).fit(skl_df,prs.reshape(prs.shape[0],-1))
	ridge_probabilities = ridge.predict(skl_df)
	ridge_probabilities =  np.divide((ridge_probabilities), np.sum(ridge_probabilities,axis=1).reshape(-1,1))
	ridge_probabilities = ridge_probabilities.reshape(ridge_probabilities.shape[0],8,8)  

	np.save('/%s/data/probabilities_pr_%s.npy'%(homedir,method),ridge_probabilities)	

def make_pr_control_jn():
	"""
	control for features of citing article
	"""
	# 1) the year of publication
	# 2) the journal in which it was published
	# 3) the number of authors
	# 4) whether the paper was a review article
	# 5) the seniority of the paper’s first and last authors.
	# 6) paper location
	# 6) paper sub-field
	df = pd.read_csv('/%s/article_data/NewArticleData2019.csv'%(homedir),header=0)
	prs = np.load('/%s/data/result_pr_df_%s.npy'%(homedir,method))
	cont = pd.read_csv('/%s/article_data/CountryAndContData.csv'%(homedir)) 
	df = df.merge(cont,how='outer',left_index=True, right_index=True)
	df = df.merge(pd.read_csv('/%s/article_data/SeniorityData.csv'%(homedir)),left_index=True, right_index=True)
	
	df = df.rename(columns={'DI':'doi'})
	df['category'] = 'none'
	sub = pd.read_csv('/%s/article_data/JoNcategories_no2019.csv'%(homedir)) 
	for cat,doi in zip(sub.category,sub.doi):
		df.iloc[np.where(df.doi==doi)[0],-1] = cat

	reg_df = pd.DataFrame(columns=['year','n_authors','journal','paper_type','senior','location','category'])
	
	for entry in tqdm.tqdm(df.iterrows(),total=len(df)):
		idx = entry[0]
		paper = entry[1]
		year = entry[1].PY
		n_authors = len(paper.AF.split(';'))
		journal = entry[1].SO
		paper_type = paper.DT
		senior = entry[1].V4
		cat = entry[1].category
		try: loc = entry[1]['FirstListed.Cont'].split()[0]
		except: loc = 'None'
		reg_df.loc[len(reg_df)] = [year,n_authors,journal,paper_type,senior,loc,cat]

	reg_df["n_authors"] = pd.to_numeric(reg_df["n_authors"])
	reg_df["year"] = pd.to_numeric(reg_df["year"])
	reg_df["senior"] = pd.to_numeric(reg_df["senior"])
	
	skl_df = pd.get_dummies(reg_df).values

	ridge = MultiOutputRegressor(RidgeCV(alphas=[1e-5,1e-4,1e-3, 1e-2, 1e-1, 1,10,25,50,75,100])).fit(skl_df,prs.reshape(prs.shape[0],-1))
	ridge_probabilities = ridge.predict(skl_df)
	ridge_probabilities =  np.divide((ridge_probabilities), np.sum(ridge_probabilities,axis=1).reshape(-1,1))
	ridge_probabilities = ridge_probabilities.reshape(ridge_probabilities.shape[0],8,8)  

	np.save('/%s/data/probabilities_pr_%s_jn.npy'%(homedir,method),ridge_probabilities)	

	df = df.rename(columns={'DI':'doi'})
	df['category'] = 'none'

def write_matrix():
	main_df = pd.read_csv('/%s/data/ArticleDataNew.csv'%(homedir))  

	prs = np.load('/%s/data/result_pr_df_%s.npy'%(homedir,method)) 

	small_matrix = np.zeros((2,2))
	matrix_idxs = {'white':0,'api':1,'hispanic':2,'black':3}
	small_idxs = {'white':0,'api':1,'hispanic':1,'black':1}

	for fa_r in ['white','api','hispanic','black']:
		for la_r in ['white','api','hispanic','black']:
			small_matrix[small_idxs[fa_r],small_idxs[la_r]] += np.sum(prs[:,matrix_idxs[fa_r],matrix_idxs[la_r]],axis=0)
	np.save('/Users/maxwell/Documents/GitHub/unbiasedciter/expected_matrix_%s.npy'%(method),np.sum(prs,axis=0))
	np.save('//Users/maxwell/Documents/GitHub/unbiasedciter/expected_small_matrix_%s.npy'%(method),small_matrix)

def convert_df():
	main_df = pd.read_csv('/%s/article_data/NewArticleData2019.csv'%(homedir),header=0)
	race_df = pd.read_csv('/%s/data/result_df_%s.csv'%(homedir,method))
	df = race_df.merge(main_df,how='outer',left_index=True, right_index=True)
	
	df['cited'] = np.nan

	for idx,paper in tqdm.tqdm(df.iterrows(),total=df.shape[0]):

		self_cites = np.array(paper.SA.split(',')).astype(int) 
		try: cites = np.array(paper.CP.split(',')).astype(int)
		except:
			if np.isnan(paper.CP):
				continue
		cites =  cites[np.isin(cites,self_cites) == False]
		df.iloc[idx,-1] = ', '.join(cites.astype(str))
	df.to_csv('/%s/article_data/NewArticleData2019_filtered.csv'%(homedir))

def make_pr_percentages(control):
	print (control)
	df = pd.read_csv('/%s/article_data/NewArticleData2019_filtered.csv'%(homedir),header=0)
	citing_prs = np.load('/%s/data/result_pr_df_%s.npy'%(homedir,method))

	if control == 'True_jn' or control == 'null_jn': 
		base_prs = np.load('/%s/data/probabilities_pr_%s_jn.npy'%(homedir,method))

	if control == True: base_prs = np.load('/%s/data/probabilities_pr_%s.npy'%(homedir,method))
	
	if control == 'null_True': base_prs = np.load('/%s/data/probabilities_pr_%s.npy'%(homedir,method))

	if control == 'null_walk' or control == 'walk':
		if walk_length == 'cited':
			base_prs = np.load('/%s/data/walk_pr_probabilities_%s_cited.npy'%(homedir,method)).reshape(-1,8,8)
		if walk_length[:3] == 'all':
			base_prs = np.load('/%s/data/walk_pr_probabilities_%s_%s.npy'%(homedir,method,walk_length)).reshape(-1,8,8)

	if type(control) != bool and control[:4] == 'null':
		matrix = np.zeros((100,df.shape[0],8,8))
		matrix[:] = np.nan

		base_matrix = np.zeros((100,df.shape[0],8,8))
		base_matrix[:] = np.nan
	else:
		matrix = np.zeros((df.shape[0],8,8))
		matrix[:] = np.nan

		base_matrix = np.zeros((df.shape[0],8,8))
		base_matrix[:] = np.nan

	if control == False:
		year_df = pd.DataFrame(columns=['year','month','prs'])
		citable_df = pd.DataFrame(columns=['year','month','index'])
		for year in df.PY.unique():
			if year < 2009:continue
			for month in df.PD.unique():
				rdf = df[(df.year<year) | ((df.year==year) & (df.PD<=month))]
				this_base_matrix = citing_prs[rdf.index.values].mean(axis=0)
				year_df = year_df.append(pd.DataFrame(np.array([year,month,this_base_matrix]).reshape(1,-1),columns=['year','month','prs']),ignore_index=True)
				citable_df = citable_df.append(pd.DataFrame(np.array([year,month,rdf.index.values]).reshape(1,-1),columns=['year','month','index']),ignore_index=True)

	if type(control) != bool and control[5:] == 'False':
		year_df = pd.DataFrame(columns=['year','month','prs'])
		citable_df = pd.DataFrame(columns=['year','month','index'])
		for year in df.PY.unique():
			if year < 2009:continue
			for month in df.PD.unique():
				rdf = df[(df.year<year) | ((df.year==year) & (df.PD<=month))]
				this_base_matrix = citing_prs[rdf.index.values].mean(axis=0)
				year_df = year_df.append(pd.DataFrame(np.array([year,month,this_base_matrix]).reshape(1,-1),columns=['year','month','prs']),ignore_index=True)
				citable_df = citable_df.append(pd.DataFrame(np.array([year,month,rdf.index.values]).reshape(1,-1),columns=['year','month','index']),ignore_index=True)




	for idx,paper in tqdm.tqdm(df.iterrows(),total=df.shape[0]):
		#only look at papers published 2009 or later
		year = paper.year
		if year < 2009:continue
		#only look at papers that cite at least 10 papers in our data
		if type(paper.cited) != str:
			if np.isnan(paper.cited)==True: continue
		n_cites = len(paper['cited'].split(','))
		if n_cites < 10: continue

		if control == 'null_True' or control == 'null_jn':
			for i in range(100):
				this_base_matrix = []
				this_matrix = [] 
				for p in base_prs[np.array(paper['cited'].split(',')).astype(int)-1]: #for each cited paper
					if np.min(p) < 0:p = p + abs(np.min(p))
					p = p + abs(np.min(p))
					p = p.flatten()/p.sum()
					this_base_matrix.append(p.reshape((8,8))) #use model prs as base matrix
					choice = np.zeros((8,8))
					choice[np.unravel_index(np.random.choice(range(64),p=p),(8,8))] = 1 #and randomly assign race category as citation matrix
					this_matrix.append(choice)
				this_base_matrix = np.sum(this_base_matrix,axis=0)
				this_matrix = np.sum(this_matrix,axis=0)
				matrix[i,idx] = this_matrix
				base_matrix[i,idx] = this_base_matrix
		elif control == 'null_False':
			citable = citable_df[(citable_df['year']==year)&(citable_df.month==paper.PD)]['index'].values[0] 
			for i in range(100):
				this_base_matrix = []
				this_matrix = []
				for p in citing_prs[np.random.choice(citable,n_cites,False)]: #for each cited paper #for naive sampling random papers
					if np.min(p) < 0:p = p + abs(np.min(p))
					p = p + abs(np.min(p))
					p = p.flatten()/p.sum()
					this_base_matrix.append(p.reshape((8,8))) #use naive base rate as base matrix
					choice = np.zeros((8,8))
					choice[np.unravel_index(np.random.choice(range(64),p=p),(8,8))] = 1 #and randomly assign race category as citation matrix based on base rates
					this_matrix.append(choice)
				this_base_matrix = np.sum(this_base_matrix,axis=0)
				this_matrix = np.sum(this_matrix,axis=0)
				matrix[i,idx] = this_matrix
				base_matrix[i,idx] = this_base_matrix
		elif control == 'null_walk':
			for i in range(100):
				this_base_matrix = []
				this_matrix = [] 
				for p in base_prs[np.array(paper['cited'].split(',')).astype(int)-1]: #for each cited paper
					choice = np.zeros((8,8))
					if np.isnan(p).any():
						this_base_matrix.append(p.reshape((8,8))) #use model prs as base matrix
						choice[:] = np.nan
						this_matrix.append(choice)
						continue
					if np.min(p) < 0:p = p + abs(np.min(p))
					p = p + abs(np.min(p))
					p = p.flatten()/p.sum()
					this_base_matrix.append(p.reshape((8,8))) #use model prs as base matrix
					choice[np.unravel_index(np.random.choice(range(64),p=p),(8,8))] = 1 #and randomly assign race category as citation matrix
					this_matrix.append(choice)
				this_base_matrix = np.nansum(this_base_matrix,axis=0)
				this_matrix = np.nansum(this_matrix,axis=0)
				matrix[i,idx] = this_matrix
				base_matrix[i,idx] = this_base_matrix
		else:
			this_matrix = citing_prs[np.array(paper['cited'].split(',')).astype(int)-1].sum(axis=0)		
			if control == False:
				this_base_matrix = year_df[(year_df.year==year) & (year_df.month<=month)]['prs'].values[0]  * n_cites
			if control == True:
				this_base_matrix = base_prs[np.array(paper['cited'].split(',')).astype(int)-1].sum(axis=0)
			if control == 'True_jn':
				this_base_matrix = base_prs[np.array(paper['cited'].split(',')).astype(int)-1].sum(axis=0)
			if control == 'walk':
				this_base_matrix = np.nansum(base_prs[np.array(paper['cited'].split(',')).astype(int)-1],axis=0)

			matrix[idx] = this_matrix
			base_matrix[idx] = this_base_matrix

	if type(control) == bool or control == 'True_jn':
		np.save('/%s/data/citation_matrix_pr_%s_%s.npy'%(homedir,method,control),matrix)
		np.save('/%s/data/base_citation_matrix_pr_%s_%s.npy'%(homedir,method,control),base_matrix)
	elif control =='null_True' or control =='null_False' or control == 'null_jn':
		np.save('/%s/data/citation_matrix_pr_%s_%s.npy'%(homedir,method,control),matrix)
		np.save('/%s/data/base_citation_matrix_pr_%s_%s.npy'%(homedir,method,control),base_matrix)
	else:
		np.save('/%s/data/citation_matrix_pr_%s_%s_%s.npy'%(homedir,method,control,walk_length),matrix)
		np.save('/%s/data/base_citation_matrix_pr_%s_%s_%s.npy'%(homedir,method,control,walk_length),base_matrix)

def self_citing(method):

	main_df = pd.read_csv('/%s/article_data/NewArticleData2019.csv'%(homedir),header=0)
	race_df = pd.read_csv('/%s/data/result_df_%s.csv'%(homedir,method))
	df = race_df.merge(main_df,how='outer',left_index=True, right_index=True)
	df['self_cites'] = np.zeros((df.shape[0]))

	for idx,paper in tqdm.tqdm(df.iterrows(),total=df.shape[0]):
		#only look at papers published 2009 or later
		year = paper.year
		if year < 2009: continue
		df.iloc[idx,-1] = len(paper.SA.split(','))
		
	scipy.stats.ttest_ind(df[(df.fa_race=='white')&(df.fa_race=='white')].self_cites,df[(df.fa_race!='white')|(df.fa_race!='white')].self_cites)
	np.median(df[(df.fa_race=='white')&(df.fa_race=='white')].self_cites.values)   
	np.median(df[(df.fa_race!='white')|(df.fa_race!='white')].self_cites.values)   
	np.mean(df[(df.fa_race=='white')&(df.fa_race=='white')].self_cites.values)   
	np.mean(df[(df.fa_race!='white')|(df.fa_race!='white')].self_cites.values)   

def plot_pr_intersections(control,citing):

	main_df = pd.read_csv('/%s/article_data/NewArticleData2019.csv'%(homedir),header=0)
	race_df = pd.read_csv('/%s/data/result_df_%s.csv'%(homedir,method.split('_')[0]))
	df = race_df.merge(main_df,how='outer',left_index=True, right_index=True)

	n_iters = 1000
	if type(control) == bool:
		matrix = np.load('/%s/data/citation_matrix_pr_%s_%s.npy'%(homedir,method,control))
		base_matrix = np.load('/%s/data/base_citation_matrix_pr_%s_%s.npy'%(homedir,method,control))
	elif control == 'all':
		matrix = np.load('/%s/data/citation_matrix_pr_%s_%s.npy'%(homedir,method,False)) 
		base_matrix = []
		for control_type in [True,False]: base_matrix.append(np.load('/%s/data/base_citation_matrix_pr_%s_%s.npy'%(homedir,method,control_type)))
		base_matrix.append(np.load('/%s/data/base_citation_matrix_pr_%s_%s_%s.npy'%(homedir,method,'walk','cited')))
		base_matrix[0] = base_matrix[0] / np.nansum(base_matrix[0]) 
		base_matrix[1] = base_matrix[1] / np.nansum(base_matrix[1]) 
		base_matrix[2] = base_matrix[2] / np.nansum(base_matrix[2]) 
		base_matrix = np.mean(base_matrix,axis=0)
	else:
		matrix = np.load('/%s/data/citation_matrix_pr_%s_%s_%s.npy'%(homedir,method,control,walk_length))
		base_matrix = np.load('/%s/data/base_citation_matrix_pr_%s_%s_%s.npy'%(homedir,method,control,walk_length))

	# np.save('/Users/maxwell/Documents/GitHub/unbiasedciter/expected_matrix_%s.npy'%(method),np.mean(matrix,axis=0))

	if type(control) == bool:
		null = np.load('/%s/data/citation_matrix_pr_%s_%s_%s.npy'%(homedir,method,'null',control))
		null_base = np.load('/%s/data/base_citation_matrix_pr_%s_%s_%s.npy'%(homedir,method,'null',control))[0]

	elif control == 'all':
		null = np.load('/%s/data/citation_matrix_pr_%s_%s_%s.npy'%(homedir,method,'null',False))
		null_base = []
		null_base.append(np.load('/%s/data/base_citation_matrix_pr_%s_%s_%s.npy'%(homedir,method,'null',True))[0])
		null_base.append(np.load('/%s/data/base_citation_matrix_pr_%s_%s_%s.npy'%(homedir,method,'null',True))[0])
		null_base.append(np.load('/%s/data/base_citation_matrix_pr_%s_%s_%s_%s.npy'%(homedir,method,'null','walk','cited'))[0])
		null_base = np.mean(null_base,axis=0)
	else:
		null = np.load('/%s/data/citation_matrix_pr_%s_%s_%s_%s.npy'%(homedir,method,'null',control,walk_length))
		null_base = np.load('/%s/data/base_citation_matrix_pr_%s_%s_%s_%s.npy'%(homedir,method,'null',control,walk_length))[0]

	boot_matrix = np.zeros((n_iters,8,8))
	boot_r_matrix = np.zeros((n_iters,8,8))
	ww_indices = df[(df.year>=2009)&(df.fa_race=='white')&(df.la_race=='white')].index
	wa_indices = df[(df.year>=2009)&(df.fa_race=='white')&(df.la_race!='white')].index
	aw_indices = df[(df.year>=2009)&(df.fa_race!='white')&(df.la_race=='white')].index
	aa_indices = df[(df.year>=2009)&(df.fa_race!='white')&(df.la_race!='white')].index

	black_indices = df[(df.year>=2009)&((df.fa_race=='black')|(df.la_race=='black'))].index
	white_indices = df[(df.year>=2009)&((df.fa_race=='white')|(df.la_race=='white'))].index
	hispanic_indices = df[(df.year>=2009)&((df.fa_race=='hispanic')|(df.la_race=='hispanic'))].index
	api_indices = df[(df.year>=2009)&((df.fa_race=='api')|(df.la_race=='api'))].index

	for b in range(n_iters):
		if citing == 'all':
			papers = np.random.choice(range(matrix.shape[0]),matrix.shape[0],replace=True)
		if citing == 'ww':
			papers = np.random.choice(ww_indices,ww_indices.shape[0],replace=True)
		if citing == 'wa':
			papers = np.random.choice(wa_indices,wa_indices.shape[0],replace=True)
		if citing == 'aw':
			papers = np.random.choice(aw_indices,aw_indices.shape[0],replace=True)
		if citing == 'aa':
			papers = np.random.choice(aa_indices,aa_indices.shape[0],replace=True)
		if citing == 'black':
			papers = np.random.choice(black_indices,black_indices.shape[0],replace=True)	
		if citing == 'hispanic':
			papers = np.random.choice(hispanic_indices,hispanic_indices.shape[0],replace=True)	
		if citing == 'api':
			papers = np.random.choice(api_indices,api_indices.shape[0],replace=True)
		if citing == 'white':
			papers = np.random.choice(white_indices,white_indices.shape[0],replace=True)		
		m = np.nansum(matrix[papers],axis=0)
		m = m / np.sum(m)
		e = np.nansum(base_matrix[papers],axis=0) 
		e = e / np.sum(e) 

		r = np.nansum(null[np.random.choice(100,1),papers],axis=0)
		r = r / np.sum(r)

		er = np.nansum(null_base[papers],axis=0) 
		er = er / np.sum(er) 


		rate = (m - e) / e
		r_rate = (r - er) / er
		boot_matrix[b] = rate
		boot_r_matrix[b] = r_rate

	# np.save('/%s/data/intersection_boot_matrix_%s.npy'%(homedir),boot_matrix,method)
		
	p_matrix = np.zeros((8,8))
	for i,j in combinations(range(8),2):
		x = boot_matrix[:,i,j]
		y = boot_r_matrix[:,i,j]
		ay = abs(y)
		ax = abs(x.mean())      
		p_matrix[i,j] = len(ay[ay>ax])  
	p_matrix = p_matrix / n_iters

	multi_mask = multipletests(p_matrix.flatten(),0.05,'holm')[0].reshape(8,8) 

	names = ['white(m)','Asian(m)','Hispanic(m)','Black(m)','white(w)','Asian(w)','Hispanic(w)','Black(w)']

	matrix_idxs = {'white(m)':0,'api(m)':1,'hispanic(m)':2,'black(m)':3,'white(w)':4,'api(w)':5,'hispanic(w)':6,'black(w)':7}

	men_aoc = np.vectorize(matrix_idxs.get)(['api(m)','hispanic(m)','black(m)'])
	women_aoc = np.vectorize(matrix_idxs.get)(['api(w)','hispanic(w)','black(w)'])

	men_aoc = boot_matrix[:,men_aoc][:,:,men_aoc].flatten()
	women_aoc = boot_matrix[:,women_aoc][:,:,women_aoc].flatten()

	white_men = np.vectorize(matrix_idxs.get)(['white(m)'])
	white_women = np.vectorize(matrix_idxs.get)(['white(w)'])

	white_men= boot_matrix[:,white_men][:,:,white_men].flatten()
	white_women = boot_matrix[:,white_women][:,:,white_women].flatten()	

	# def exact_mc_perm_test(xs, ys, nmc=10000):
	# 	n, k = len(xs), 0
	# 	diff = np.abs(np.mean(xs) - np.mean(ys))
	# 	zs = np.concatenate([xs, ys])
	# 	for j in range(nmc):
	# 		np.random.shuffle(zs)
	# 		k += diff <= np.abs(np.mean(zs[:n]) - np.mean(zs[n:]))
	# 	return k / nmc

	# p = exact_mc_perm_test(men_aoc,women_aoc) 
	# p = log_p_value(p)

	def direction(d):
	 	if d<=0: return 'less'
	 	else: return 'greater'

	diff = (men_aoc-women_aoc)
	high,low = np.percentile(diff,97.5),np.percentile(diff,2.5)
	low,high = np.around(low*100,2),np.around(high*100,2)
	diff = np.around(diff.mean()*100,2)
	print (control)
	if control == 'walk': print (walk_length)
	print ('AoC men papers are cited at %s percentage points %s than women AoC papers 95pecentCI=%s,%s'%(abs(diff),direction(diff),low,high))

	diff = (white_men-men_aoc[:len(white_men)])
	high,low = np.percentile(diff,97.5),np.percentile(diff,2.5)
	low,high = np.around(low*100,2),np.around(high*100,2)
	diff = np.around(diff.mean()*100,2)
	if control == 'walk': print (walk_length)
	print ('white men papers are cited at %s percentage points %s than  men AoC papers 95pecentCI=%s,%s'%(abs(diff),direction(diff),low,high))

	diff = (white_men-white_women)
	high,low = np.percentile(diff,97.5),np.percentile(diff,2.5)
	low,high = np.around(low*100,2),np.around(high*100,2)
	diff = np.around(diff.mean()*100,2)
	print ('white men papers are cited at %s percentage points %s than white women papers 95pecentCI=%s,%s'%(abs(diff),direction(diff),low,high))

	diff = (white_women-women_aoc[:len(white_women)])
	high,low = np.percentile(diff,97.5),np.percentile(diff,2.5)
	low,high = np.around(low*100,2),np.around(high*100,2)
	diff = np.around(diff.mean()*100,2)
	print ('white women papers are cited at %s percentage points %s than women-AoC papers 95pecentCI=%s,%s'%(abs(diff),direction(diff),low,high))

	diff = (white_women-men_aoc[:len(white_women)])
	high,low = np.percentile(diff,97.5),np.percentile(diff,2.5)
	low,high = np.around(low*100,2),np.around(high*100,2)
	diff = np.around(diff.mean()*100,2)
	if control == 'walk': print (walk_length)
	print ('white women papers are cited at %s percentage points %s than men AoC papers 95pecentCI=%s,%s'%(abs(diff),direction(diff),low,high))
	# 1/0

	if type(control) == bool:
		orig_matrix = np.load('/%s/data/citation_matrix_pr_%s_%s.npy'%(homedir,method,control))
		orig_base_matrix = np.load('/%s/data/base_citation_matrix_pr_%s_%s.npy'%(homedir,method,control))
	elif control == 'all':
		orig_matrix = np.load('/%s/data/citation_matrix_pr_%s_%s.npy'%(homedir,method,False)) 
		orig_base_matrix = []
		for control_type in [True,False]:
			orig_base_matrix.append(np.load('/%s/data/base_citation_matrix_pr_%s_%s.npy'%(homedir,method,control_type)))
		orig_base_matrix.append(np.load('/%s/data/base_citation_matrix_pr_%s_%s_%s.npy'%(homedir,method,'walk','cited')))
		orig_base_matrix[0] = orig_base_matrix[0] / np.nansum(orig_base_matrix[0]) 
		orig_base_matrix[1] = orig_base_matrix[1] / np.nansum(orig_base_matrix[1]) 
		orig_base_matrix[2] = orig_base_matrix[2] / np.nansum(orig_base_matrix[2]) 
		orig_base_matrix = np.mean(orig_base_matrix,axis=0)
	else:
		orig_matrix = np.load('/%s/data/citation_matrix_pr_%s_%s_%s.npy'%(homedir,method,control,walk_length))
		orig_base_matrix = np.load('/%s/data/base_citation_matrix_pr_%s_%s_%s.npy'%(homedir,method,control,walk_length))


	matrix_idxs = {'white_M':0,'api_M':1,'hispanic_M':2,'black_M':3,'white_W':4,'api_W':5,'hispanic_W':6,'black_W':7}


	df = pd.DataFrame(columns=['bias type','bias amount','boot','race'])
	for race in ['white','black','api','hispanic']:
		for idx in range(n_iters):
			#norm matrix
			if citing == 'all':
				pick = np.random.choice(np.arange(orig_matrix.shape[0]),int(orig_matrix.shape[0]),replace=True)
				# papers = np.random.choice(range(matrix.shape[0]),matrix.shape[0],replace=True)
			if citing == 'ww':
				pick = np.random.choice(ww_indices,ww_indices.shape[0],replace=True)
			if citing == 'wa':
				pick = np.random.choice(wa_indices,wa_indices.shape[0],replace=True)
			if citing == 'aw':
				pick = np.random.choice(aw_indices,aw_indices.shape[0],replace=True)
			if citing == 'aa':
				pick = np.random.choice(aa_indices,aa_indices.shape[0],replace=True)
			if citing == 'black':
				pick = np.random.choice(black_indices,black_indices.shape[0],replace=True)
			if citing == 'hispanic':
				pick = np.random.choice(hispanic_indices,hispanic_indices.shape[0],replace=True)
			if citing == 'api':
				pick = np.random.choice(api_indices,api_indices.shape[0],replace=True)
			if citing == 'white':
				pick = np.random.choice(white_indices,white_indices.shape[0],replace=True)
			matrix = orig_matrix[pick]
			matrix = matrix / np.nansum(matrix)
			base_matrix= orig_base_matrix[pick]
			base_matrix = base_matrix / np.nansum(base_matrix)


			man_e1 = np.nansum(matrix[:,matrix_idxs['%s_M'%(race)],matrix_idxs['%s_M'%(race)]])
			man_b1 = np.nansum(base_matrix[:,matrix_idxs['%s_M'%(race)],matrix_idxs['%s_M'%(race)]])
			woman_e1 = np.nansum(matrix[:,matrix_idxs['%s_W'%(race)],matrix_idxs['%s_W'%(race)]])
			woman_b1 = np.nansum(base_matrix[:,matrix_idxs['%s_W'%(race)],matrix_idxs['%s_W'%(race)]])


			x =  ((man_e1 - man_b1)/ man_b1)  - ((woman_e1 - woman_b1)/ woman_b1)  # bias against women within this race

			if race == 'black':
				groups = [np.vectorize(matrix_idxs.get)(['white_M','white_W']),
				np.vectorize(matrix_idxs.get)(['black_M','black_W'])]

			if race == 'api':
				groups = [np.vectorize(matrix_idxs.get)(['white_M','white_W']),
				np.vectorize(matrix_idxs.get)(['api_M','api_W'])]

			if race == 'hispanic':
				groups = [np.vectorize(matrix_idxs.get)(['white_M','white_W']),
				np.vectorize(matrix_idxs.get)(['hispanic_M','hispanic_W'])]

			if race == 'white':
				groups = [np.vectorize(matrix_idxs.get)(['hispanic_M','hispanic_W','api_M','api_W','black_M','black_W']),
				np.vectorize(matrix_idxs.get)(['white_M','white_W'])]


			race_e1 = np.nansum(matrix[:,groups[1],groups[1]])
			race_b1 = np.nansum(base_matrix[:,groups[1],groups[1]])


			other_e1 = np.nansum(matrix[:,groups[0],groups[0]])

			other_b1 = np.nansum(base_matrix[:,groups[0],groups[0]])

			other = (other_e1 - other_b1) / other_b1
			race_c = (race_e1 - race_b1) / race_b1

			y = other - race_c # bias against this race
			df = df.append(pd.DataFrame(np.array(['gender',x,idx,race]).reshape(1,4),columns=['bias type','bias amount','boot','race']),ignore_index=True)
			df = df.append(pd.DataFrame(np.array(['race',y,idx,race]).reshape(1,4),columns=['bias type','bias amount','boot','race']),ignore_index=True)
	
	df['bias amount'] = df['bias amount'].astype(float) *  100
	df.race[df.race == 'hispanic'] = 'Hispanic' 
	df.race[df.race == 'api'] = 'Asian' 
	df.race[df.race == 'black'] = 'Black' 

	plt.close()
	sns.set(style='white',font='Palatino')
	cmap = sns.diverging_palette(220, 10, as_cmap=True)
	fig = plt.figure(figsize=(7.5,3),constrained_layout=True)
	gs = gridspec.GridSpec(2, 2, figure=fig)
	ax1 = fig.add_subplot(gs[:2,:1])
	ax2 = fig.add_subplot(gs[:2,1:])
	plt.sca(ax1)
	d = np.around(np.nanmean(boot_matrix,axis=0)*100,0)
	# d[multi_mask==False] = np.nan
	heat = sns.heatmap(d,annot=True,fmt='g',cmap=cmap,vmin=-25,vmax=25,annot_kws={"size": 8})
	heat.set_ylabel('first author',labelpad=0)
	heat.set_yticklabels(names,rotation=25)
	heat.set_xlabel('last author',labelpad=0)  
	heat.set_xticklabels(names,rotation=65)
	heat.set_title('a',{'fontweight':'bold'},'left',pad=1)


	for text, show_annot in zip(ax1.texts, (element for row in multi_mask for element in row)):
		text.set_visible(show_annot)
	cbar = heat.collections[0].colorbar
	cbar.ax.set_yticklabels(["{:.0%}".format(i/100) for i in cbar.get_ticks()])	
	
	plt.sca(ax2)
	df['bias amount'] = df['bias amount'].astype(float)*-1
	sns.barplot(data=df,y='bias amount',x='race',hue='bias type',ci='sd',palette=['grey','white'],order=['white','Asian','Hispanic','Black'],**{'edgecolor':'grey'})
	plt.ylabel('percentage point difference')

	ax2.yaxis.set_major_locator(plt.MaxNLocator(8))
	ax2.tick_params(axis='y', which='major', pad=-5)
	ax2.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
	ax2.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0)) 
	plt.legend(ncol=2,fontsize='small',frameon=False,columnspacing=0.5,handletextpad=0)
	plt.title('b',{'fontweight':'bold'},'left',pad=1)
	if type(control) == bool: plt.savefig('/%s/figures/intersection/intersection_%s_%s_%s.pdf'%(homedir,method,control,citing))
	else: plt.savefig('/%s/figures/intersection/intersection_%s_%s_%s_%s.pdf'%(homedir,method,control,walk_length,citing)) 
	plt.close()

def all_inter():
	for c in ['aa','aw','wa','ww']:
		plot_pr_intersections(control,c)

def all_inter_main():
	global walk_length
	plot_pr_intersections(False,'all')
	plot_pr_intersections(True,'all')
	walk_length = 'cited'
	plot_pr_intersections('walk','all')
	walk_length = 'all'
	plot_pr_intersections('walk','all')

def plot_ethnicolor_confusion():

	# order = [] 
	# for r in wiki_2_race.keys():
	# 	order.append(r.split(',')[-1])
	# r = [[873, 44, 7, 6, 6, 114, 8, 10, 7, 1, 8, 9, 6],
	#  [17, 1300, 7, 20, 2, 58, 7, 6, 2, 0, 36, 10, 2],
	#  [10, 10, 1188, 23, 107, 121, 21, 22, 15, 9, 17, 22, 7],
	#  [5, 18, 48, 321, 72, 126, 12, 32, 31, 6, 37, 21, 5],
	#  [6, 3, 118, 36, 824, 80, 45, 64, 23, 6, 15, 16, 12],
	#  [52, 11, 57, 45, 52, 7341, 45, 260, 161, 39, 59, 101, 66],
	#  [8, 5, 16, 14, 19, 84, 1262, 122, 21, 44, 18, 30, 23],
	#  [7, 8, 27, 20, 66, 633, 119, 881, 59, 71, 80, 45, 32],
	#  [13, 7, 14, 32, 34, 488, 37, 112, 1417, 41, 125, 118, 21],
	#  [3, 0, 5, 7, 5, 167, 19, 98, 36, 318, 26, 23, 67],
	#  [12, 12, 16, 19, 16, 174, 23, 56, 64, 18, 1437, 213, 22],
	#  [4, 10, 13, 25, 8, 165, 34, 39, 99, 24, 147, 1790, 16],
	#  [10, 2, 3, 7, 13, 141, 30, 31, 18, 44, 13, 11, 640]]

	name_dict = {'asian':[0,1,2],'black':[3,4],'white':[5,6,7,8,9,11,12],'hispanic':[10]}

	names = ['asian','black','hispanic','white']
	# small_r = np.zeros((4,4))
	# r = np.array(r)
	# for idx,i in enumerate(names):
	# 	for jdx,j in enumerate(names):
	# 		small_r[idx,jdx] = r[name_dict[i],:][:,name_dict[j]].sum()


	small_r = [[2214,363,257,1693],[82 , 13522  ,  427  , 4409],[   144  ,  408 , 12410 , 15624],[   438  , 3511 ,  3804, 138256]]

	plt.close()
	sns.set(style='white',font='Palatino')
	fig = plt.figure(figsize=(7.5,4))
	ax1 = fig.add_subplot(121)
	ax2 = fig.add_subplot(122)
	# ax1 = fig.add_subplot(gs[:,0])
	# ax2 = fig.add_subplot(gs[:,1])
	plt.sca(ax1)

	order = ['Asian','Black','Hispanic','white']
	# asian, hispanic, black, white
	# small_r[[0,2,1,3]]
	heat = sns.heatmap(np.array(small_r)[[0,2,1,3]][:,[0,2,1,3]],vmax=20000,annot=True,fmt='g',annot_kws={"size": 10},cbar_kws={'shrink': .5},square=True)
	locs, labels = plt.yticks()  
	plt.yticks(locs,order,rotation=360,**{'fontsize':10},) 
	locs, labels = plt.xticks() 
	plt.xticks(locs,order,rotation=90,**{'fontsize':10}) 
	plt.ylabel('observed racial category',**{'fontsize':12}) 
	plt.xlabel('predicted racial category',**{'fontsize':12}) 
	plt.title('a',{'fontweight':'bold'},'left',pad=3)
	plt.tight_layout()

	plt.sca(ax2)
	r = [[5743, 42, 796, 3490],[257, 1693, 218, 22649],[173,82,25118,7609],[694,1157, 2442, 27837]]
	order = ['Asian','Black','Hispanic','white']
	heat = sns.heatmap(np.array(r),vmax=20000,annot=True,fmt='g',annot_kws={"size": 10},cbar_kws={'shrink': .5},square=True)
	locs, labels = plt.yticks()  
	plt.yticks(locs,order,rotation=360,**{'fontsize':10},) 
	locs, labels = plt.xticks() 
	plt.xticks(locs,order,rotation=90,**{'fontsize':10}) 
	plt.ylabel('observed racial category',**{'fontsize':12}) 
	plt.xlabel('predicted racial category',**{'fontsize':12}) 
	plt.title('b',{'fontweight':'bold'},'left',pad=3)
	plt.tight_layout()

	plt.savefig('/%s/dazed_and_confused.pdf'%(homedir))
	plt.savefig('/%s/dazed_and_confused.png'%(homedir))

def plot_histy():
	
	control,within_poc,walk_papers = func_vars[0],func_vars[1],func_vars[2]
	"""
	Figure 2
	"""
	n_iters = 10000
	main_df = pd.read_csv('/%s/article_data/NewArticleData2019.csv'%(homedir),header=0)
	race_df = pd.read_csv('/%s/data/result_df_%s.csv'%(homedir,method))
	df = race_df.merge(main_df,how='outer',left_index=True, right_index=True)
	

	if control == False:
		null = np.load('/%s/data/citation_matrix_pr_%s_%s.npy'%(homedir,method,'null_False'))
		null_base = np.load('/%s/data/base_citation_matrix_pr_%s_%s.npy'%(homedir,method,'null_False'))
	if control == True:
		null = np.load('/%s/data/citation_matrix_pr_%s_%s.npy'%(homedir,method,'null_True'))
		null_base = np.load('/%s/data/base_citation_matrix_pr_%s_%s.npy'%(homedir,method,'null_True'))
	if control == 'walk':
		null = np.load('/%s/data/citation_matrix_pr_%s_%s_%s.npy'%(homedir,method,'null_walk',walk_length))
		null_base = np.load('/%s/data/base_citation_matrix_pr_%s_%s_%s.npy'%(homedir,method,'null_walk',walk_length))

	if type(control) == bool:
		matrix = np.load('/%s/data/citation_matrix_pr_%s_%s.npy'%(homedir,method,control))
		base_matrix = np.load('/%s/data/base_citation_matrix_pr_%s_%s.npy'%(homedir,method,control))
	else:
		matrix = np.load('/%s/data/citation_matrix_pr_%s_%s_%s.npy'%(homedir,method,control,walk_length))
		base_matrix = np.load('/%s/data/base_citation_matrix_pr_%s_%s_%s.npy'%(homedir,method,control,walk_length))

	if walk_papers == True:
		walk_base_matrix = np.load('/%s/data/base_citation_matrix_%s_walk.npy'%(homedir,method))
		matrix[np.isnan(walk_base_matrix[:,0,0])] = np.nan
		base_matrix[np.isnan(walk_base_matrix[:,0,0])] = np.nan

	matrix_idxs = {'white_M':0,'api_M':1,'hispanic_M':2,'black_M':3,'white_W':4,'api_W':5,'hispanic_W':6,'black_W':7}

	if within_poc == False:
		groups = [np.vectorize(matrix_idxs.get)(['white_M','white_W',]),
		np.vectorize(matrix_idxs.get)(['api_M','api_W','hispanic_M','hispanic_W','black_M','black_W',])]
		names = ['white-white','white-poc','poc-white','poc-poc']

	plot_matrix = np.zeros((matrix.shape[0],len(groups),len(groups)))
	plot_base_matrix = np.zeros((matrix.shape[0],len(groups),len(groups)))
	plot_null = np.zeros((null.shape[0],matrix.shape[0],len(groups),len(groups)))
	plot_null_base = np.zeros((null.shape[0],matrix.shape[0],len(groups),len(groups)))

	for i in range(len(groups)):
		for j in range(len(groups)):
			plot_matrix[:,i,j] = np.nansum(matrix[:,groups[i]][:,:,groups[j]].reshape(matrix.shape[0],-1),axis=1)
			plot_base_matrix[:,i,j] = np.nansum(base_matrix[:,groups[i]][:,:,groups[j]].reshape(matrix.shape[0],-1),axis=1)
			for iteration in range(null.shape[0]):
				plot_null[iteration,:,i,j] = np.nansum(null[iteration,:,groups[i]][:,:,groups[j]].reshape(matrix.shape[0],-1),axis=1)
				plot_null_base[iteration,:,i,j] = np.nansum(null_base[iteration,:,groups[i]][:,:,groups[j]].reshape(matrix.shape[0],-1),axis=1)


	#make sure that, if we don't have data for a paper, we also are not including it's base rates
	#this is mostly for when the random walk fails because it's not part of the graph.
	x = plot_matrix.sum(axis=1).sum(axis=1)
	y = plot_base_matrix.sum(axis=1).sum(axis=1)
	mask = np.where(x==0)[0] 
	assert y[mask].sum() == 0

	data_type = np.zeros((4)).astype(str)
	data_type[:] = 'real'
	rdata_type = np.zeros((4)).astype(str)
	rdata_type[:] = 'random'


	data = []
	papers = df[df.year>=2009]
	for boot in range(n_iters):
		boot_papers = papers.sample(len(papers),replace=True).index

		emperical = plot_matrix[boot_papers].reshape(-1,4)[:,3]
		total = plot_matrix[boot_papers].reshape(-1,4).sum(axis=1) 
		# emperical = emperical / total
		data.append(emperical)

	plt.close()
	sns.set(style='white',font='Palatino')
	fig = plt.figure(figsize=(7.5,3),constrained_layout=True)
	gs = gridspec.GridSpec(12, 10, figure=fig)
	ax1 = fig.add_subplot(gs[:12,:5])
	plt.sca(ax1)	
	dp = sns.distplot(np.nanmean(data,axis=0), kde=False, rug=False)
	# plt.xlim(0,.2) 
	plt.xlim(0.5,2)

	ax2 = fig.add_subplot(gs[0:6,5:])
	ax3 = fig.add_subplot(gs[6:,5:])
	
	data = []
	papers = df[(df.year>=2009)&(df.fa_race=='white')&(df.la_race=='white')]
	for boot in range(n_iters):
		boot_papers = papers.sample(len(papers),replace=True).index

		emperical = plot_matrix[boot_papers].reshape(-1,4)[:,3]
		total = plot_matrix[boot_papers].reshape(-1,4).sum(axis=1) 
		# emperical = emperical / total
		data.append(emperical)
	

	plt.sca(ax2)	
	dp = sns.distplot(np.nanmean(data,axis=0), kde=False, rug=False)
	# plt.xlim(0,.2) 
	plt.xlim(0.5,2)


	
	data = []
	papers = df[(df.year>=2009)&((df.fa_race!='white')| (df.la_race!='white'))]
	for boot in range(n_iters):
		boot_papers = papers.sample(len(papers),replace=True).index

		emperical = plot_matrix[boot_papers].reshape(-1,4)[:,3]
		total = plot_matrix[boot_papers].reshape(-1,4).sum(axis=1) 
		# emperical = emperical / total
		data.append(emperical)
	
	plt.sca(ax3)
	dp = sns.distplot(np.nanmean(data,axis=0), kde=False, rug=False)
	# plt.xlim(0,.2) 
	plt.xlim(0.5,2)

def plot_compare():
	names = ['white-white','white-AoC','AoC-white','AoC-AoC']
	jn_df = pd.read_csv('/%s/%s_%s_2compare.csv'%(homedir,'True_jn',True))
	jn_df['model'] = 'subfield'
	df = pd.read_csv('/%s/%s_%s_2compare.csv'%(homedir,True,True))
	df['model'] = 'paper'
	df = df.append(jn_df,ignore_index=True)


	plt.close()
	sns.set(style='white',font='Palatino')
	fig = plt.figure(figsize=(7.5,3.5),constrained_layout=True)
	gs = gridspec.GridSpec(2,2, figure=fig)
	ax0 = fig.add_subplot(gs[:,0])
	ax1 = fig.add_subplot(gs[0,1])
	ax2 = fig.add_subplot(gs[1,1])


	axes = [ax0,ax1,ax2]


	for ax,citation_type in zip(axes,['all','white','AoC']):
		if citation_type == 'all': plot_df = df[df['citing authors'] == 'all']
		if citation_type == 'white': plot_df = df[df['citing authors'] == 'white']
		if citation_type == 'AoC': plot_df = df[df['citing authors'] == 'AoC']


		plt.sca(ax)


		bx = sns.violinplot(data=plot_df,y='citation_rate',x='citation_type',hue='model',split=True,palette=pal,order=names,saturation=1,cut=0,scale='width')
		for index,violin in enumerate([bx.collections[:3],bx.collections[3:6],bx.collections[6:9],bx.collections[9:]]):
			i,j,k = violin
			i.set_color(pal[index])
			j.set_color(pal[index])
			k.set_color('white')


		# plt.ylabel("percent over-/under-citation",labelpad=0)
		plt.xlabel('')
		plt.ylabel('')
		plt.title('%s'%(citation_type),{'fontweight':'bold'},'left',pad=0)
		ax.yaxis.set_major_locator(plt.MaxNLocator(8))
		ax.tick_params(axis='y', which='major', pad=-5)
		ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
		ax.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))

		mean = plot_df.groupby('citation_type',sort=False).mean()['citation_rate']
		std = plot_df.groupby('citation_type',sort=False).std()['citation_rate']
		maxval = plot_df.groupby('citation_type',sort=False).max()['citation_rate']
		minval = plot_df.groupby('citation_type',sort=False).min()['citation_rate']	
		for i,citation_type in enumerate([ 'white-white','white-AoC','AoC-white','AoC-AoC']):
			y = plot_df[(plot_df.model=='paper')&(plot_df.citation_type==citation_type)].citation_rate.values
			x = plot_df[(plot_df.model=='subfield')&(plot_df.citation_type==citation_type)].citation_rate.values
			diff=x-y
			ci_mean = np.around(np.mean(diff),2)
			ci = [np.around(np.percentile(diff,2.5),2),np.around(np.percentile(diff, 97.5),2)]
			m,s = mean.values[i],std.values[i]
			if m > 0: loc = minval.values[i] - s
			if m < 0: loc = maxval.values[i] + s

			if loc > plt.ylim()[1]:
				loc = plt.ylim()[1]
			if loc < plt.ylim()[0]:
				loc = plt.ylim()[0]
			ax.text(i,loc,'%s<%s>%s'%(ci[0],ci_mean,ci[1]),horizontalalignment='center',fontsize=8)
		ax.legend_.remove() 
		plt.savefig('/%s/figures/percentages/jneuro_papers.pdf'%(homedir))

def plot_pr_percentages_booty_matrix_jn(func_vars):
	
	control,within_poc,walk_papers = func_vars[0],func_vars[1],func_vars[2]
	"""
	Figure 2
	"""
	n_iters = 100
	main_df = pd.read_csv('/%s/article_data/NewArticleData2019.csv'%(homedir),header=0)
	race_df = pd.read_csv('/%s/data/result_df_%s.csv'%(homedir,method))
	df = race_df.merge(main_df,how='outer',left_index=True, right_index=True)
	
	df = df.rename(columns={'DI':'doi'})
	df['category'] = 'none'

	sub = pd.read_csv('/%s/article_data/JoNcategories_no2019.csv'%(homedir)) 
	for cat,doi in zip(sub.category,sub.doi):
		df.iloc[np.where(df.doi==doi)[0],-1] = cat

	if control == False:
		null = np.load('/%s/data/citation_matrix_pr_%s_%s.npy'%(homedir,method,'null_False'))
		null_base = np.load('/%s/data/base_citation_matrix_pr_%s_%s.npy'%(homedir,method,'null_False'))
	if control == True:
		null = np.load('/%s/data/citation_matrix_pr_%s_%s.npy'%(homedir,method,'null_True'))
		null_base = np.load('/%s/data/base_citation_matrix_pr_%s_%s.npy'%(homedir,method,'null_True'))
	if control == 'walk':
		null = np.load('/%s/data/citation_matrix_pr_%s_%s_%s.npy'%(homedir,method,'null_walk',walk_length))
		null_base = np.load('/%s/data/base_citation_matrix_pr_%s_%s_%s.npy'%(homedir,method,'null_walk',walk_length))

	if type(control) == bool:
		matrix = np.load('/%s/data/citation_matrix_pr_%s_%s.npy'%(homedir,method,control))
		base_matrix = np.load('/%s/data/base_citation_matrix_pr_%s_%s.npy'%(homedir,method,control))
	else:
		matrix = np.load('/%s/data/citation_matrix_pr_%s_%s_%s.npy'%(homedir,method,control,walk_length))
		base_matrix = np.load('/%s/data/base_citation_matrix_pr_%s_%s_%s.npy'%(homedir,method,control,walk_length))

	if walk_papers == True:
		walk_base_matrix = np.load('/%s/data/base_citation_matrix_%s_walk.npy'%(homedir,method))
		matrix[np.isnan(walk_base_matrix[:,0,0])] = np.nan
		base_matrix[np.isnan(walk_base_matrix[:,0,0])] = np.nan

	matrix_idxs = {'white_M':0,'api_M':1,'hispanic_M':2,'black_M':3,'white_W':4,'api_W':5,'hispanic_W':6,'black_W':7}

	if within_poc == False:
		groups = [np.vectorize(matrix_idxs.get)(['white_M','white_W',]),
		np.vectorize(matrix_idxs.get)(['api_M','api_W','hispanic_M','hispanic_W','black_M','black_W',])]
		names = ['white-white','white-AoC','AoC-white','AoC-AoC']

	if within_poc == 'black':
		# groups = [np.vectorize(matrix_idxs.get)(['white_M','white_W','api_M','api_W','hispanic_M','hispanic_W',]),
		groups = [np.vectorize(matrix_idxs.get)(['white_M','white_W']),
		np.vectorize(matrix_idxs.get)(['black_M','black_W',])]
		names = ['white-white','white-black','black-white','black-black']

	if within_poc == 'api':
		# groups = [np.vectorize(matrix_idxs.get)(['white_M','white_W','hispanic_M','hispanic_W','black_M','black_W',]),
		groups = [np.vectorize(matrix_idxs.get)(['white_M','white_W']),
		np.vectorize(matrix_idxs.get)(['api_M','api_W',])]
		names = ['white-white','white-asian','asian-white','asian-asian']

	if within_poc == 'hispanic':
		groups = [np.vectorize(matrix_idxs.get)(['white_M','white_W']),
		# groups = [np.vectorize(matrix_idxs.get)(['white_M','white_W','api_M','api_W','black_M','black_W',]),
		np.vectorize(matrix_idxs.get)(['hispanic_M','hispanic_W',])]
		names = ['white-white','white-hispanic','hispanic-white','hispanic-hispanic']


	plot_matrix = np.zeros((matrix.shape[0],len(groups),len(groups)))
	plot_base_matrix = np.zeros((matrix.shape[0],len(groups),len(groups)))
	plot_null = np.zeros((null.shape[0],matrix.shape[0],len(groups),len(groups)))
	plot_null_base = np.zeros((null.shape[0],matrix.shape[0],len(groups),len(groups)))

	for i in range(len(groups)):
		for j in range(len(groups)):
			plot_matrix[:,i,j] = np.nansum(matrix[:,groups[i]][:,:,groups[j]].reshape(matrix.shape[0],-1),axis=1)
			plot_base_matrix[:,i,j] = np.nansum(base_matrix[:,groups[i]][:,:,groups[j]].reshape(matrix.shape[0],-1),axis=1)
			for iteration in range(null.shape[0]):
				plot_null[iteration,:,i,j] = np.nansum(null[iteration,:,groups[i]][:,:,groups[j]].reshape(matrix.shape[0],-1),axis=1)
				plot_null_base[iteration,:,i,j] = np.nansum(null_base[iteration,:,groups[i]][:,:,groups[j]].reshape(matrix.shape[0],-1),axis=1)


	#make sure that, if we don't have data for a paper, we also are not including it's base rates
	#this is mostly for when the random walk fails because it's not part of the graph.
	x = plot_matrix.sum(axis=1).sum(axis=1)
	y = plot_base_matrix.sum(axis=1).sum(axis=1)
	mask = np.where(x==0)[0] 
	assert y[mask].sum() == 0


	plt.close()
	sns.set(style='white',font='Palatino')
	fig = plt.figure(figsize=(7.5,7.5),constrained_layout=True)
	gs = gridspec.GridSpec(4,2, figure=fig)
	ax0 = fig.add_subplot(gs[0,0])
	ax1 = fig.add_subplot(gs[1,0])
	ax2 = fig.add_subplot(gs[2,0])
	ax3 = fig.add_subplot(gs[3,0])
	ax4 = fig.add_subplot(gs[0,1])
	ax5 = fig.add_subplot(gs[1,1])
	ax6 = fig.add_subplot(gs[2,1])
	ax7 = fig.add_subplot(gs[3,1])

	axes = [ax0,ax1,ax2,ax3,ax4,ax5,ax6,ax7]
	categories = ['all','behavioral/cognitive','behavioral/systems/cognitive', 'brief communications', 'cellular/molecular', 'development/plasticity/repair','neurobiology of disease', 'systems/circuits']

	for thisax,cat in zip(axes,categories):
		plt.sca(thisax)
		data_type = np.zeros((4)).astype(str)
		data_type[:] = 'real'
		rdata_type = np.zeros((4)).astype(str)
		rdata_type[:] = 'random'

		data = pd.DataFrame(columns=['citation_rate','citation_type','data_type'])
		if cat == 'all': papers = df[(df.journal=='JOURNAL OF NEUROSCIENCE')&(df.year>=2009)]
		else: papers = df[(df.category==cat)&(df.year>=2009)]
		for boot in range(n_iters):
			boot_papers = papers.sample(len(papers),replace=True).index
			
			emperical = np.nanmean(plot_matrix[boot_papers],axis=0)		
			expected = np.nanmean(plot_base_matrix[boot_papers],axis=0)
			emperical = emperical / np.sum(emperical)
			expected = expected / np.sum(expected)
			rate = (emperical - expected) / expected
		

			random = np.nanmean(plot_null[np.random.choice(plot_null.shape[0]),boot_papers],axis=0)
			e_random = np.nanmean(plot_null_base[np.random.choice(plot_null.shape[0]),boot_papers],axis=0)
			random = random / np.sum(random)
			e_random = e_random / np.sum(e_random)
			r_rate = (random - e_random) / e_random

			data = data.append(pd.DataFrame(data= np.array([rate.flatten(),names,data_type]).swapaxes(0,1),columns=['citation_rate','citation_type','data_type']),ignore_index=True)
			data = data.append(pd.DataFrame(data= np.array([r_rate.flatten(),names,rdata_type]).swapaxes(0,1),columns=['citation_rate','citation_type','data_type']),ignore_index=True)   
		
		data.citation_rate = (data.citation_rate.astype(float)*100)
		p_vals = np.zeros((4))
		for idx,name in enumerate(names):
			x = data[(data.data_type=='real')&(data.citation_type==name)].citation_rate.values
			y = data[(data.data_type=='random')&(data.citation_type==name)].citation_rate.values
			ay = abs(y)
			ax = abs(x.mean())
			p_vals[idx] = len(ay[ay>ax])     
		
		p_vals = p_vals / n_iters

		if type(control) == bool:
			data.to_csv('/%s/data/citaion_rates_%s_%s.csv'%(homedir,method,control),index=False)
		if control == 'walk': data.to_csv('/%s/data/citaion_rates_%s_%s_%s.csv'%(homedir,method,control,walk_length),index=False)

		plot_data = data[data.data_type=='real']
		mean = plot_data.groupby('citation_type',sort=False).mean()
		std = plot_data.groupby('citation_type',sort=False).std()	


		bx = sns.violinplot(data=plot_data,y='citation_rate',x='citation_type',palette=pal,order=names,saturation=1,cut=0,scale='width')
		for i,v in enumerate(bx.collections[::2]):
			v.set_color(pal[i])
		bx2 = sns.violinplot(data=data[data.data_type=='random'],y='citation_rate',x='citation_type',palette=pal,order=names,saturation=1,cut=0,scale='width',inner=None)
		for i,v in enumerate(bx2.collections[8:]):
			v.set_color([pal[i][0],pal[i][1],pal[i][2],.35])
		plt.ylabel(" ",labelpad=0)
		plt.xlabel('')
		plt.title('%s,n=%s'%(cat,len(papers)),{'fontweight':'bold'},'left',pad=1)
		thisax.yaxis.set_major_locator(plt.MaxNLocator(8))
		thisax.tick_params(axis='y', which='major', pad=-5)
		thisax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
		thisax.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
		for i in range(4):
			m,s = mean.values[i],std.values[i]
			loc = m + (s*3)
			low = np.around(m - (s*2),1)[0]
			high = np.around(m + (s*2),1)[0]
			m = np.around(m,1)[0]
			if m > 0: loc = loc * -1
			if loc > plt.ylim()[1]:
				loc = plt.ylim()[1]
			if loc < plt.ylim()[0]:
				loc = plt.ylim()[0]
			thisax.text(i,loc,'%s<%s>%s\n%s'%(low,m,high,log_p_value(p_vals[i])),horizontalalignment='center',fontsize=8)
		plt.savefig('sub_fields.pdf') 

def plot_pr_percentages_booty_matrix(func_vars):
	
	control,within_poc,jneuro_papers = func_vars[0],func_vars[1],func_vars[2]
	"""
	Figure 2
	"""
	n_iters = 100
	t_n_iters = 10
	main_df = pd.read_csv('/%s/article_data/NewArticleData2019.csv'%(homedir),header=0)
	race_df = pd.read_csv('/%s/data/result_df_%s.csv'%(homedir,method))
	df = race_df.merge(main_df,how='outer',left_index=True, right_index=True)

	if control == False:
		null = np.load('/%s/data/citation_matrix_pr_%s_%s.npy'%(homedir,method,'null_False'))
		null_base = np.load('/%s/data/base_citation_matrix_pr_%s_%s.npy'%(homedir,method,'null_False'))
	if control == True:
		null = np.load('/%s/data/citation_matrix_pr_%s_%s.npy'%(homedir,method,'null_True'))
		null_base = np.load('/%s/data/base_citation_matrix_pr_%s_%s.npy'%(homedir,method,'null_True'))
	if control == 'True_jn':
		null = np.load('/%s/data/citation_matrix_pr_%s_%s.npy'%(homedir,method,'null_jn'))
		null_base = np.load('/%s/data/base_citation_matrix_pr_%s_%s.npy'%(homedir,method,'null_jn'))
	if control == 'walk':
		null = np.load('/%s/data/citation_matrix_pr_%s_%s_%s.npy'%(homedir,method,'null_walk',walk_length))
		null_base = np.load('/%s/data/base_citation_matrix_pr_%s_%s_%s.npy'%(homedir,method,'null_walk',walk_length))

	if type(control) == bool or control == 'True_jn':
		matrix = np.load('/%s/data/citation_matrix_pr_%s_%s.npy'%(homedir,method,control))
		base_matrix = np.load('/%s/data/base_citation_matrix_pr_%s_%s.npy'%(homedir,method,control))
	else:
		matrix = np.load('/%s/data/citation_matrix_pr_%s_%s_%s.npy'%(homedir,method,control,walk_length))
		base_matrix = np.load('/%s/data/base_citation_matrix_pr_%s_%s_%s.npy'%(homedir,method,control,walk_length))

	# if jneuro_papers == True:
	# 	walk_base_matrix = np.load('/%s/data/base_citation_matrix_%s_walk.npy'%(homedir,method))
	# 	matrix[np.isnan(walk_base_matrix[:,0,0])] = np.nan
	# 	base_matrix[np.isnan(walk_base_matrix[:,0,0])] = np.nan

	matrix_idxs = {'white_M':0,'api_M':1,'hispanic_M':2,'black_M':3,'white_W':4,'api_W':5,'hispanic_W':6,'black_W':7}

	if within_poc == False:
		groups = [np.vectorize(matrix_idxs.get)(['white_M','white_W',]),
		np.vectorize(matrix_idxs.get)(['api_M','api_W','hispanic_M','hispanic_W','black_M','black_W',])]
		names = ['white-white','white-AoC','AoC-white','AoC-AoC']

	if within_poc == 'black':
		# groups = [np.vectorize(matrix_idxs.get)(['white_M','white_W','api_M','api_W','hispanic_M','hispanic_W',]),
		groups = [np.vectorize(matrix_idxs.get)(['white_M','white_W']),
		np.vectorize(matrix_idxs.get)(['black_M','black_W',])]
		names = ['white-white','white-black','black-white','black-black']

	if within_poc == 'api':
		# groups = [np.vectorize(matrix_idxs.get)(['white_M','white_W','hispanic_M','hispanic_W','black_M','black_W',]),
		groups = [np.vectorize(matrix_idxs.get)(['white_M','white_W']),
		np.vectorize(matrix_idxs.get)(['api_M','api_W',])]
		names = ['white-white','white-asian','asian-white','asian-asian']

	if within_poc == 'hispanic':
		groups = [np.vectorize(matrix_idxs.get)(['white_M','white_W']),
		# groups = [np.vectorize(matrix_idxs.get)(['white_M','white_W','api_M','api_W','black_M','black_W',]),
		np.vectorize(matrix_idxs.get)(['hispanic_M','hispanic_W',])]
		names = ['white-white','white-hispanic','hispanic-white','hispanic-hispanic']


	plot_matrix = np.zeros((matrix.shape[0],len(groups),len(groups)))
	plot_base_matrix = np.zeros((matrix.shape[0],len(groups),len(groups)))
	plot_null = np.zeros((null.shape[0],matrix.shape[0],len(groups),len(groups)))
	plot_null_base = np.zeros((null.shape[0],matrix.shape[0],len(groups),len(groups)))

	for i in range(len(groups)):
		for j in range(len(groups)):
			plot_matrix[:,i,j] = np.nansum(matrix[:,groups[i]][:,:,groups[j]].reshape(matrix.shape[0],-1),axis=1)
			plot_base_matrix[:,i,j] = np.nansum(base_matrix[:,groups[i]][:,:,groups[j]].reshape(matrix.shape[0],-1),axis=1)
			for iteration in range(null.shape[0]):
				plot_null[iteration,:,i,j] = np.nansum(null[iteration,:,groups[i]][:,:,groups[j]].reshape(matrix.shape[0],-1),axis=1)
				plot_null_base[iteration,:,i,j] = np.nansum(null_base[iteration,:,groups[i]][:,:,groups[j]].reshape(matrix.shape[0],-1),axis=1)


	#make sure that, if we don't have data for a paper, we also are not including it's base rates
	#this is mostly for when the random walk fails because it's not part of the graph.
	x = plot_matrix.sum(axis=1).sum(axis=1)
	y = plot_base_matrix.sum(axis=1).sum(axis=1)
	mask = np.where(x==0)[0] 
	assert y[mask].sum() == 0


	for papers in [df[df.year>=2009],df[(df.year>=2009)&(df.fa_race=='white')&(df.la_race=='white')],df[(df.year>=2009)&((df.fa_race!='white')|(df.la_race!='white'))]]:
		print (papers.citation_count.sum())
		sum_cites = papers.citation_count.sum()
		papers = papers.index
		emperical = np.nanmean(plot_matrix[papers],axis=0)
		expected = np.nanmean(plot_base_matrix[papers],axis=0)
		emperical = emperical / np.sum(emperical)
		expected = expected / np.sum(expected)
		rate = (emperical - expected) / expected
		p = np.array([np.around(emperical.flatten()*100,1),np.around(expected.flatten()*100,1)]).flatten()
		print ('Of the citations given between 2009 and 2019, WW papers received %s, compared to %s for WA papers, %s for AW papers, and %s for AA papers. The expected proportions based on the pool of citable papers were %s for WW, %s for WA, %s for AW, and %s for AA.'%(p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7]))
		p = np.around(rate.flatten()*100,1)
		print ('By this measure, WW papers were cited %s more than expected, WA papers were cited %s less than expected, AW papers were cited %s less than expected, and AA papers were cited %s less than expected.'%(p[0],p[1],p[2],p[3]))
		p = np.around(((emperical - expected) * sum_cites).flatten(),-1).astype(int)
		print ('These values correspond to WW papers being cited roughly %s more times than expected, compared to roughly %s more times for WA papers, %s fewer for AW papers, and %s fewer for AA papers'%(p[0],p[1],p[2],p[3]))

	data_type = np.zeros((4)).astype(str)
	data_type[:] = 'real'
	rdata_type = np.zeros((4)).astype(str)
	rdata_type[:] = 'random'


	data = pd.DataFrame(columns=['citation_rate','citation_type','data_type'])
	papers = df[df.year>=2009]
	if jneuro_papers == True: papers = papers[papers.journal=='JOURNAL OF NEUROSCIENCE']
	for boot in range(n_iters):
		boot_papers = papers.sample(len(papers),replace=True).index
		
		emperical = np.nanmean(plot_matrix[boot_papers],axis=0)		
		expected = np.nanmean(plot_base_matrix[boot_papers],axis=0)
		emperical = emperical / np.sum(emperical)
		expected = expected / np.sum(expected)
		rate = (emperical - expected) / expected
		


		random = np.nanmean(plot_null[np.random.choice(plot_null.shape[0]),boot_papers],axis=0)
		e_random = np.nanmean(plot_null_base[np.random.choice(plot_null.shape[0]),boot_papers],axis=0)
		random = random / np.sum(random)
		e_random = e_random / np.sum(e_random)
		r_rate = (random - e_random) / e_random

		data = data.append(pd.DataFrame(data= np.array([rate.flatten(),names,data_type]).swapaxes(0,1),columns=['citation_rate','citation_type','data_type']),ignore_index=True)
		data = data.append(pd.DataFrame(data= np.array([r_rate.flatten(),names,rdata_type]).swapaxes(0,1),columns=['citation_rate','citation_type','data_type']),ignore_index=True)   
	
	data.citation_rate = (data.citation_rate.astype(float)*100)
	p_vals = np.zeros((4))
	for idx,name in enumerate(names):
		x = data[(data.data_type=='real')&(data.citation_type==name)].citation_rate.values
		y = data[(data.data_type=='random')&(data.citation_type==name)].citation_rate.values
		ay = abs(y)
		ax = abs(x.mean())
		p_vals[idx] = len(ay[ay>ax])    
	
	p_vals = p_vals / n_iters

	if type(control) == bool:
		data.to_csv('/%s/data/citaion_rates_%s_%s.csv'%(homedir,method,control),index=False)
	if control == 'walk': data.to_csv('/%s/data/citaion_rates_%s_%s_%s.csv'%(homedir,method,control,walk_length),index=False)

	plot_data = data[data.data_type=='real']
	mean = plot_data.groupby('citation_type',sort=False).mean()
	std = plot_data.groupby('citation_type',sort=False).std()	
	all_data = plot_data.copy()
	all_data['citing authors'] = 'all'
	plt.close()
	sns.set(style='white',font='Palatino')
	fig = plt.figure(figsize=(7.5,3),constrained_layout=True)
	gs = gridspec.GridSpec(12, 10, figure=fig)
	ax1 = fig.add_subplot(gs[:12,:5])
	plt.sca(ax1)	
	bx = sns.violinplot(data=plot_data,y='citation_rate',x='citation_type',palette=pal,order=names,saturation=1,cut=0,scale='width')
	for i,v in enumerate(bx.collections[::2]):
		v.set_color(pal[i])
	bx2 = sns.violinplot(data=data[data.data_type=='random'],y='citation_rate',x='citation_type',palette=pal,order=names,saturation=1,cut=0,scale='width',inner=None)
	for i,v in enumerate(bx2.collections[8:]):
		v.set_color([pal[i][0],pal[i][1],pal[i][2],.35])
	plt.ylabel("percent over-/under-citation",labelpad=0)
	plt.xlabel('')
	plt.title('a, all citers',{'fontweight':'bold'},'left',pad=1)
	ax1.yaxis.set_major_locator(plt.MaxNLocator(8))
	ax1.tick_params(axis='y', which='major', pad=-5)
	ax1.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
	ax1.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
	for i in range(4):
		m,s = mean.values[i],std.values[i]
		loc = m + (s*3)
		low = np.around(m - (s*2),1)[0]
		high = np.around(m + (s*2),1)[0]
		m = np.around(m,1)[0]
		if m > 0: loc = loc * -1
		if loc > plt.ylim()[1]:
			loc = plt.ylim()[1]
		if loc < plt.ylim()[0]:
			loc = plt.ylim()[0]
		ax1.text(i,loc,'%s<%s>%s\n%s'%(low,m,high,log_p_value(p_vals[i])),horizontalalignment='center',fontsize=8)
	
	ax2 = fig.add_subplot(gs[0:6,5:])
	ax3 = fig.add_subplot(gs[6:,5:])
	
	plt.sca(ax2)


	data = pd.DataFrame(columns=['citation_rate','citation_type','data_type'])
	papers = df[(df.year>=2009)&(df.fa_race=='white')&(df.la_race=='white')]
	if jneuro_papers == True: papers = papers[papers.journal=='JOURNAL OF NEUROSCIENCE']
	for boot in range(n_iters):
		boot_papers = papers.sample(len(papers),replace=True).index
		
		emperical = np.nanmean(plot_matrix[boot_papers],axis=0)		
		expected = np.nanmean(plot_base_matrix[boot_papers],axis=0)
		emperical = emperical / np.sum(emperical)
		expected = expected / np.sum(expected)
		rate = (emperical - expected) / expected
		


		random = np.nanmean(plot_null[np.random.choice(plot_null.shape[0]),boot_papers],axis=0)
		e_random = np.nanmean(plot_null_base[np.random.choice(plot_null.shape[0]),boot_papers],axis=0)
		random = random / np.sum(random)
		e_random = e_random / np.sum(e_random)
		r_rate = (random - e_random) / e_random


		data = data.append(pd.DataFrame(data= np.array([rate.flatten(),names,data_type]).swapaxes(0,1),columns=['citation_rate','citation_type','data_type']),ignore_index=True)
		data = data.append(pd.DataFrame(data= np.array([r_rate.flatten(),names,rdata_type]).swapaxes(0,1),columns=['citation_rate','citation_type','data_type']),ignore_index=True)   
	
	data.citation_rate = (data.citation_rate.astype(float)*100)
	p_vals = np.zeros((4))
	for idx,name in enumerate(names):
		x = data[(data.data_type=='real')&(data.citation_type==name)].citation_rate.values
		y = data[(data.data_type=='random')&(data.citation_type==name)].citation_rate.values
		ay = abs(y)
		ax = abs(x.mean())
		p_vals[idx] = len(ay[ay>ax])                              
	p_vals = p_vals / n_iters


	
	plot_data = data[data.data_type=='real']
	plot_data['citing authors'] = 'white'
	all_data = all_data.append(plot_data,ignore_index=True)

	
	mean = plot_data.groupby('citation_type',sort=False).mean()
	std = plot_data.groupby('citation_type',sort=False).std()	

	plt.sca(ax2)	
	bx = sns.violinplot(data=plot_data,y='citation_rate',x='citation_type',palette=pal,order=names,saturation=1,cut=0,scale='width')
	for i,v in enumerate(bx.collections[::2]):
		v.set_color(pal[i])
	bx2 = sns.violinplot(data=data[data.data_type=='random'],y='citation_rate',x='citation_type',palette=pal,order=names,saturation=1,cut=0,scale='width',inner=None)
	for i,v in enumerate(bx2.collections[8:]):
		v.set_color([pal[i][0],pal[i][1],pal[i][2],.35])
	# plt.ylabel("percent over-/under-citation",labelpad=0)
	plt.xlabel('')
	plt.ylabel('')
	plt.title('b, white citers',{'fontweight':'bold'},'left',pad=1)
	ax2.yaxis.set_major_locator(plt.MaxNLocator(8))
	ax2.tick_params(axis='y', which='major', pad=-5)
	ax2.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
	ax2.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
	for i in range(4):
		m,s = mean.values[i],std.values[i]
		loc = m + (s*3)
		low = np.around(m - (s*2),1)[0]
		high = np.around(m + (s*2),1)[0]
		m = np.around(m,1)[0]
		if m > 0: loc = loc * -1
		if loc > plt.ylim()[1]:
			loc = plt.ylim()[1]
		if loc < plt.ylim()[0]:
			loc = plt.ylim()[0]
		ax2.text(i,loc,'%s<%s>%s\n%s'%(low,m,high,log_p_value(p_vals[i])),horizontalalignment='center',fontsize=8)
	
	plt.sca(ax3)


	data = pd.DataFrame(columns=['citation_rate','citation_type','data_type'])
	papers = df[(df.year>=2009)&((df.fa_race!='white')|(df.la_race!='white'))]
	if jneuro_papers == True: papers = papers[papers.journal=='JOURNAL OF NEUROSCIENCE']
	for boot in range(n_iters):
		boot_papers = papers.sample(len(papers),replace=True).index
		
		emperical = np.nanmean(plot_matrix[boot_papers],axis=0)		
		expected = np.nanmean(plot_base_matrix[boot_papers],axis=0)
		emperical = emperical / np.sum(emperical)
		expected = expected / np.sum(expected)
		rate = (emperical - expected) / expected
		


		random = np.nanmean(plot_null[np.random.choice(plot_null.shape[0]),boot_papers],axis=0)
		e_random = np.nanmean(plot_null_base[np.random.choice(plot_null.shape[0]),boot_papers],axis=0)
		random = random / np.sum(random)
		e_random = e_random / np.sum(e_random)
		r_rate = (random - e_random) / e_random

		data = data.append(pd.DataFrame(data= np.array([rate.flatten(),names,data_type]).swapaxes(0,1),columns=['citation_rate','citation_type','data_type']),ignore_index=True)
		data = data.append(pd.DataFrame(data= np.array([r_rate.flatten(),names,rdata_type]).swapaxes(0,1),columns=['citation_rate','citation_type','data_type']),ignore_index=True)   
	
	data.citation_rate = (data.citation_rate.astype(float)*100)
	p_vals = np.zeros((4))
	for idx,name in enumerate(names):
		x = data[(data.data_type=='real')&(data.citation_type==name)].citation_rate.values
		y = data[(data.data_type=='random')&(data.citation_type==name)].citation_rate.values
		ay = abs(y)
		ax = abs(x.mean())
		p_vals[idx] = len(ay[ay>ax])                          
	p_vals = p_vals / n_iters


	plot_data = data[data.data_type=='real']
	plot_data['citing authors'] = 'AoC'
	all_data = all_data.append(plot_data,ignore_index=True)
	if method == 'florida':all_data.to_csv('/%s/%s_%s_2compare.csv'%(homedir,control,jneuro_papers))
	
	mean = plot_data.groupby('citation_type',sort=False).mean()
	std = plot_data.groupby('citation_type',sort=False).std()	
	
	bx = sns.violinplot(data=plot_data,y='citation_rate',x='citation_type',palette=pal,order=names,saturation=1,cut=0,scale='width')
	for i,v in enumerate(bx.collections[::2]):
		v.set_color(pal[i])
	bx2 = sns.violinplot(data=data[data.data_type=='random'],y='citation_rate',x='citation_type',palette=pal,order=names,saturation=1,cut=0,scale='width',inner=None)
	for i,v in enumerate(bx2.collections[8:]):
		v.set_color([pal[i][0],pal[i][1],pal[i][2],.35])
	# plt.ylabel("percent over-/under-citation",labelpad=0)
	plt.xlabel('')
	plt.ylabel('')
	plt.title('c, citers of color',{'fontweight':'bold'},'left',pad=1)
	ax3.yaxis.set_major_locator(plt.MaxNLocator(8))
	ax3.tick_params(axis='y', which='major', pad=-5)
	ax3.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
	ax3.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
	for i in range(4):
		m,s = mean.values[i],std.values[i]
		loc = m + (s*3)
		low = np.around(m - (s*2),1)[0]
		high = np.around(m + (s*2),1)[0]
		m = np.around(m,1)[0]
		if m > 0: loc = loc * -1
		if loc > plt.ylim()[1]:
			loc = plt.ylim()[1]
		if loc < plt.ylim()[0]:
			loc = plt.ylim()[0]
		ax3.text(i,loc,'%s<%s>%s\n%s'%(low,m,high,log_p_value(p_vals[i])),horizontalalignment='center',fontsize=8)


	ylim = np.array([ax3.get_ylim(),ax2.get_ylim()]).min(),np.array([ax3.get_ylim(),ax2.get_ylim()]).max()
	plt.sca(ax3)
	plt.ylim(ylim)
	plt.sca(ax2)
	plt.ylim(ylim)

	if type(control) == bool or control == 'True_jn': plt.savefig('/%s/figures/percentages/method-%s_control-%s_poc-%s_wp-%s.pdf'%(homedir,method,control,within_poc,jneuro_papers))
	else: plt.savefig('/%s/figures/percentages/method-%s_control-%s_poc-%s_wl-%s.pdf'%(homedir,method,control,within_poc,walk_length))
	plt.close()

	

	# return None
	"""
	temporal trends
	"""
	n_iters = t_n_iters
	white_data = pd.DataFrame(columns=['citation_rate','citation_type','year','base_rate','emperical_rate','data_type','boot'])
	for year in range(2009,2020):
		papers = df[(df.year==year)&(df.fa_race=='white')&(df.la_race=='white')]
		for boot in range(n_iters):
			boot_papers = papers.sample(len(papers),replace=True).index

			emperical = np.nanmean(plot_matrix[boot_papers],axis=0)
			expected = np.nanmean(plot_base_matrix[boot_papers],axis=0)
			emperical = emperical / np.sum(emperical)
			expected = expected / np.sum(expected)
			rate = (emperical - expected) / expected

			random = np.nanmean(plot_null[np.random.choice(plot_null.shape[0]),boot_papers],axis=0)
			e_random = np.nanmean(plot_null_base[np.random.choice(plot_null.shape[0]),boot_papers],axis=0)
			random = random / np.sum(random)
			e_random = e_random / np.sum(e_random)
			r_rate = (random - e_random) / e_random

			boot_df = pd.DataFrame(data= np.array([rate.flatten(),names]).swapaxes(0,1),columns=['citation_rate','citation_type'])
			boot_df['year'] = year
			boot_df['base_rate'] = expected.flatten()
			boot_df['emperical_rate'] = emperical.flatten()
			boot_df['data_type'] = 'real'
			boot_df['boot'] = boot
			white_data = white_data.append(boot_df,ignore_index=True)   

			boot_df = pd.DataFrame(data= np.array([r_rate.flatten(),names]).swapaxes(0,1),columns=['citation_rate','citation_type'])
			boot_df['year'] = year
			boot_df['base_rate'] = e_random.flatten()
			boot_df['emperical_rate'] = random.flatten()
			boot_df['data_type'] = 'random'
			boot_df['boot'] = boot
			white_data = white_data.append(boot_df,ignore_index=True)   
		
	white_data = white_data.dropna()
	white_data.citation_rate = (white_data.citation_rate.astype(float)*100)
	white_data.base_rate = (white_data.base_rate .astype(float)*100)
	white_data.emperical_rate = (white_data.emperical_rate.astype(float)*100)

	slope_boot_df = pd.DataFrame(columns=['slope','data','citation_type'])
	for boot in range(n_iters):
		for name in names:
			real_slope = scipy.stats.linregress(white_data[(white_data.data_type=='real')&(white_data.citation_type==name)&(white_data.boot==boot)].citation_rate.values,range(11))[0] 
			random_slope = scipy.stats.linregress(white_data[(white_data.data_type=='random')&(white_data.citation_type==name)&(white_data.boot==boot)].citation_rate.values,range(11))[0] 
			slope_boot_df = slope_boot_df.append(pd.DataFrame(data= np.array([[real_slope,random_slope],['real','random'],[name,name]]).swapaxes(0,1),columns=['slope','data','citation_type']))

	slope_boot_df.slope=slope_boot_df.slope.astype(float)


	non_white_data = pd.DataFrame(columns=['citation_rate','citation_type','year','base_rate','emperical_rate','data_type','boot'])
	for year in range(2009,2020):
			papers = df[(df.year==year)&((df.fa_race!='white')|(df.la_race!='white'))]
			for boot in range(n_iters):
				boot_papers = papers.sample(len(papers),replace=True).index

				emperical = np.nanmean(plot_matrix[boot_papers],axis=0)
				expected = np.nanmean(plot_base_matrix[boot_papers],axis=0)
				emperical = emperical / np.sum(emperical)
				expected = expected / np.sum(expected)
				rate = (emperical - expected) / expected

				random = np.nanmean(plot_null[np.random.choice(plot_null.shape[0]),boot_papers],axis=0)
				e_random = np.nanmean(plot_null_base[np.random.choice(plot_null.shape[0]),boot_papers],axis=0)
				random = random / np.sum(random)
				e_random = e_random / np.sum(e_random)
				r_rate = (random - e_random) / e_random

				boot_df = pd.DataFrame(data= np.array([rate.flatten(),names]).swapaxes(0,1),columns=['citation_rate','citation_type'])
				boot_df['year'] = year
				boot_df['base_rate'] = expected.flatten()
				boot_df['emperical_rate'] = emperical.flatten()
				boot_df['data_type'] = 'real'
				boot_df['boot'] = boot
				non_white_data = non_white_data.append(boot_df,ignore_index=True)   

				boot_df = pd.DataFrame(data= np.array([r_rate.flatten(),names]).swapaxes(0,1),columns=['citation_rate','citation_type'])
				boot_df['year'] = year
				boot_df['base_rate'] = e_random.flatten()
				boot_df['emperical_rate'] = random.flatten()
				boot_df['data_type'] = 'random'
				boot_df['boot'] = boot
				non_white_data = non_white_data.append(boot_df,ignore_index=True)   
		
	non_white_data = non_white_data.dropna()
	non_white_data.citation_rate = (non_white_data.citation_rate.astype(float)*100)
	non_white_data.base_rate = (non_white_data.base_rate .astype(float)*100)
	non_white_data.emperical_rate = (non_white_data.emperical_rate.astype(float)*100)

	non_white_slope_boot_df = pd.DataFrame(columns=['slope','data','citation_type'])
	for boot in range(n_iters):
		for name in names:
			real_slope = scipy.stats.linregress(non_white_data[(non_white_data.data_type=='real')&(non_white_data.citation_type==name)&(non_white_data.boot==boot)].citation_rate.values,range(11))[0] 
			random_slope = scipy.stats.linregress(non_white_data[(non_white_data.data_type=='random')&(non_white_data.citation_type==name)&(non_white_data.boot==boot)].citation_rate.values,range(11))[0] 
			non_white_slope_boot_df = non_white_slope_boot_df.append(pd.DataFrame(data= np.array([[real_slope,random_slope],['real','random'],[name,name]]).swapaxes(0,1),columns=['slope','data','citation_type']))

	non_white_slope_boot_df.slope=non_white_slope_boot_df.slope.astype(float)

	plt.close()
	sns.set(style='white',font='Palatino')
	fig = plt.figure(figsize=(7.5,6),constrained_layout=True)
	gs = fig.add_gridspec(4, 4)
	
	ax1 = fig.add_subplot(gs[:2,:2])
	ax2 = fig.add_subplot(gs[:2,2:])

	ax3 = fig.add_subplot(gs[2,0])
	ax4 = fig.add_subplot(gs[2,1])
	ax5 = fig.add_subplot(gs[3,0])
	ax6 = fig.add_subplot(gs[3,1])

	ax7 = fig.add_subplot(gs[2,2])
	ax8 = fig.add_subplot(gs[2,3])
	ax9 = fig.add_subplot(gs[3,2])
	ax10 = fig.add_subplot(gs[3,3])

	plt.sca(ax1)
	sns.lineplot(x="year", y="citation_rate",hue="citation_type",data=white_data[white_data.data_type=='real'],ax=ax1,hue_order=names,ci='sd',palette=pal)
	plt.legend(labels=names,ncol=2,fontsize='small',frameon=False,columnspacing=0.5,handletextpad=0)#bbox_to_anchor=(0., 1.05))
	ax1.set_xlabel('')
	plt.title('a, white citers',{'fontweight':'bold'},'left',pad=1)
	ax1.set_ylabel('percent over-/under-citation',labelpad=0)
	ax1.tick_params(axis='x', which='major', pad=-5)
	ax1.yaxis.set_major_locator(plt.MaxNLocator(8))
	ax1.tick_params(axis='y', which='major', pad=-5)
	ax1.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
	ax1.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
	ax1.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
	plt.axhline(0, color="grey", clip_on=False,linestyle='--')
	plt.xlim(2009,2019) 
	for color,name in zip(pal,names):
		y_val=white_data[(white_data.data_type=='real')&(white_data.citation_type==name)&((white_data.year==2017)|(white_data.year==2018)|(white_data.year==2019))].citation_rate.max()
		x = slope_boot_df[(slope_boot_df.data=='real')&(slope_boot_df.citation_type==name)].slope.values
		y = slope_boot_df[(slope_boot_df.data=='random')&(slope_boot_df.citation_type==name)].slope.values
		p_val = min(len(y[y>x.mean()]),len(y[y<x.mean()]))  
		p_val = p_val/n_iters
		print (p_val)
		p_val = log_p_value(p_val)
		plt.text(2019,y_val,'slope=%s,%s'%(np.around(x.mean(),2),p_val),horizontalalignment='right',verticalalignment='bottom',fontsize=8,color=color)



	plt.sca(ax2)
	sns.lineplot(x="year", y="citation_rate",hue="citation_type",data=non_white_data[non_white_data.data_type=='real'],ax=ax2,hue_order=names,ci='sd',palette=pal)
	plt.legend(labels=names,ncol=2,fontsize='small',frameon=False,columnspacing=0.5,handletextpad=0)#,bbox_to_anchor=(0., 1.05))
	ax2.set_xlabel('')
	# plt.axhline(0, color="grey", clip_on=False,axes=ax2,linestyle='--')
	plt.title('b, citer of color',{'fontweight':'bold'},'left',pad=1)
	sns.despine()
	ax2.set_ylabel('percent over-/under-citation',labelpad=0)
	ax2.tick_params(axis='x', which='major', pad=-5)
	ax2.yaxis.set_major_locator(plt.MaxNLocator(8))
	ax2.tick_params(axis='y', which='major', pad=-5)
	ax2.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
	ax2.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
	ax2.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
	fig.text(0.00, 0.26, 'percentage of citations', va='center', rotation='vertical')
	plt.axhline(0, color="grey", clip_on=False,linestyle='--')   
	plt.xlim(2009,2019) 
	for color,name in zip(pal,names):
		y_val=non_white_data[(non_white_data.data_type=='real')&(non_white_data.citation_type==name)&((non_white_data.year==2017)|(non_white_data.year==2018)|(non_white_data.year==2019))].citation_rate.max()
		x = non_white_slope_boot_df[(non_white_slope_boot_df.data=='real')&(non_white_slope_boot_df.citation_type==name)].slope.values
		y = non_white_slope_boot_df[(non_white_slope_boot_df.data=='random')&(non_white_slope_boot_df.citation_type==name)].slope.values
		p_val = min(len(y[y>x.mean()]),len(y[y<x.mean()]))  
		p_val = p_val/n_iters
		print (p_val)
		p_val = log_p_value(p_val)
		plt.text(2019,y_val,'slope=%s,%s'%(np.around(x.mean(),2),p_val),horizontalalignment='right',verticalalignment='bottom',fontsize=8,color=color)


	ylim = np.array(np.array([ax1.get_ylim(),ax1.get_ylim()]).min(),np.array([ax2.get_ylim(),ax2.get_ylim()]).max())
	plt.sca(ax1)
	plt.ylim(ylim*1.1)
	plt.sca(ax2)
	plt.ylim(ylim*1.1)

	white_data = white_data[white_data.data_type=='real']
	non_white_data = non_white_data[non_white_data.data_type=='real']

	label = True


	white_max = np.max([white_data.groupby('citation_type').max()['emperical_rate'],white_data.groupby('citation_type').max()['base_rate']],axis=0)
	white_min = np.min([white_data.groupby('citation_type').min()['emperical_rate'],white_data.groupby('citation_type').min()['base_rate']],axis=0)

	aoc_max = np.max([non_white_data.groupby('citation_type').max()['emperical_rate'],non_white_data.groupby('citation_type').max()['base_rate']],axis=0)
	aoc_min = np.min([non_white_data.groupby('citation_type').min()['emperical_rate'],non_white_data.groupby('citation_type').min()['base_rate']],axis=0)

	min_y = np.flip(np.min([white_min,aoc_min],axis=0))
	max_y = np.flip(np.max([white_max,aoc_max],axis=0))

	i = 0
	for ax,citation_type,color in zip([ax3,ax4,ax5,ax6],white_data.citation_type.unique(),pal):
		plt.sca(ax)
		ax.clear()
		if label == True:
			plt.title('c, white citers',{'fontweight':'bold'},'left',pad=1)
			label = False
		tmp_ax0 = sns.lineplot(x="year", y="emperical_rate",data=white_data[white_data.citation_type==citation_type],ci='sd',color=color,marker='o')
		tmp_ax1 = sns.lineplot(x="year", y="base_rate",data=white_data[white_data.citation_type==citation_type],ci='sd',color='grey',marker='o')
		if citation_type == 'white-white' or citation_type== 'poc-poc' :
			s,i_,r,p,std = scipy.stats.linregress(white_data[white_data.citation_type==citation_type].groupby('year').mean()['emperical_rate'],range(11))
			print (s,p)
			s,i_,r,p,std = scipy.stats.linregress(white_data[white_data.citation_type==citation_type].groupby('year').mean()['base_rate'],range(11))
			print (s,p)
		ax.set_xlabel('')
		# ax3.set_ylabel('percentage of citations',labelpad=0)
		sns.despine()
		ax.yaxis.set_major_locator(plt.MaxNLocator(6))
		ax.tick_params(axis='y', which='major', pad=-5)
		ax.tick_params(axis='x', which='major', bottom=False,top=False,labelbottom=False)
		ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
		ax.set_ylabel('')
		ax.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=1)) 
		plt.ylim(min_y[i],max_y[i])
		i = i + 1

	label = True
	i = 0
	for ax,citation_type,color in zip([ax7,ax8,ax9,ax10],non_white_data.citation_type.unique(),pal):
		plt.sca(ax)
		if label == True: 
			plt.title('d, citers of color',{'fontweight':'bold'},'left',pad=1)
			label = False
		tmp_ax0 = sns.lineplot(x="year", y="emperical_rate",data=non_white_data[non_white_data.citation_type==citation_type],ci='sd',color=color,marker='o')
		tmp_ax1 = sns.lineplot(x="year", y="base_rate",data=non_white_data[non_white_data.citation_type==citation_type],ci='sd',color='grey',marker='o')
		ax.set_xlabel('')
		# ax3.set_ylabel('percentage of citations',labelpad=0)
		sns.despine()
		ax.yaxis.set_major_locator(plt.MaxNLocator(6))
		ax.tick_params(axis='y', which='major', pad=-5)
		ax.tick_params(axis='x', which='major', bottom=False,top=False,labelbottom=False)
		ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
		ax.set_ylabel('')
		ax.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=1)) 
		plt.ylim(min_y[i],max_y[i])
		i = i + 1
	
	if type(control) == bool: plt.savefig('/%s/figures/temporal/method-%s_control-%s_poc-%s_wp-%s.pdf'%(homedir,method,control,within_poc,walk_papers))
	else: plt.savefig('/%s/figures/temporal/method-%s_control-%s_poc-%s_wl-%s.pdf'%(homedir,method,control,within_poc,walk_length))
	plt.close()

def compare_nulls():


	paper_data = pd.read_csv('/%s/data/citaion_rates_%s_%s.csv'%(homedir,method,'True'))
	paper_data = paper_data[paper_data.data_type=='real']
	walk_cite = pd.read_csv('/%s/data/citaion_rates_%s_%s_%s.csv'%(homedir,method,'walk','cited'))
	walk_cite = walk_cite[walk_cite.data_type=='real']
	walk_all = pd.read_csv('/%s/data/citaion_rates_%s_%s_%s.csv'%(homedir,method,'walk','all'))
	walk_all = walk_all[walk_all.data_type=='real']
	raw_data = pd.read_csv('/%s/data/citaion_rates_%s_%s.csv'%(homedir,method,'False'))
	raw_data = raw_data[raw_data.data_type=='real']

	for citation_type in ['white-white','poc-white','white-poc','poc-poc']:
		x = paper_data[paper_data.citation_type==citation_type].citation_rate.values
		y = walk_cite[walk_cite.citation_type==citation_type].citation_rate.values
		print (scipy.stats.ttest_rel(x,y),x.mean(),y.mean())
	
	for citation_type in ['white-white','poc-white','white-poc','poc-poc']:
		x = paper_data[paper_data.citation_type==citation_type].citation_rate.values
		y = walk_all[walk_all.citation_type==citation_type].citation_rate.values
		print (scipy.stats.ttest_rel(x,y),x.mean(),y.mean())

	for citation_type in ['white-white','poc-white','white-poc','poc-poc']:
		x = raw_data[raw_data.citation_type==citation_type].citation_rate.values
		y = walk_all[walk_all.citation_type==citation_type].citation_rate.values
		print (scipy.stats.ttest_rel(x,y),x.mean(),y.mean())

	for citation_type in ['white-white','poc-white','white-poc','poc-poc']:
		x = raw_data[raw_data.citation_type==citation_type].citation_rate.values
		y = walk_cite[walk_cite.citation_type==citation_type].citation_rate.values
		print (scipy.stats.ttest_rel(x,y),x.mean(),y.mean())

def plot_all():
	global walk_length
	plot_pr_percentages_booty_matrix([False,False,False])
	plot_pr_percentages_booty_matrix([True,False,False])
	plot_pr_intersections(False,'all')
	plot_pr_intersections(True,'all')

	walk_length = 'all'
	plot_pr_percentages_booty_matrix(['walk',False,False])
	plot_pr_intersections('walk','all')
	walk_length = 'cited'
	plot_pr_percentages_booty_matrix(['walk',False,False])
	plot_pr_intersections('walk','all')

def make_networks():
	"""
	co-author network, authors are nodes, co-authorship is an edge
	"""
	author_df = pd.read_csv('/%s/article_data/NewArticleData2019.csv'%(homedir),header=0)

	g = igraph.Graph()
	node_2_a = {}
	a_2_node = {}
	a_2_paper = {}
	node_idx = 0
	#add authors to graph
	edges = []
	years = []
	for p in tqdm.tqdm(author_df.iterrows(),total=len(author_df)):
		#get authors of this paper
		authors = p[1].AF.split('; ')
		# p_races = p[1].races.split('_')
		for ai,a in enumerate(authors):
			#authors on more than one paper, so skip if already in graph
			if a in a_2_node.keys():continue
			# store papers for each author, used for edges later
			if a in a_2_paper.keys(): a_2_paper[a.strip()] = a_2_paper[a.strip()].append(p[0])
			else: a_2_paper[a.strip()] = [p[0]]
			#store index and author name
			node_2_a[node_idx] = a.strip()
			a_2_node[a.strip()] = node_idx
			#add author and race
			g.add_vertex(node_idx)
			# races.append(p_races[ai])
			#on to the next node/author
			node_idx += 1
		#add edges between authors
		year = p[1].PY
		for co_a_i in p[1].AF.split('; '):
			for co_a_j in p[1].AF.split('; '):
				if co_a_j == co_a_i: continue
				#look up nodal_index of co_author
				# 1/0
				nodes = [a_2_node[co_a_i.strip()],a_2_node[co_a_j.strip()]]
				i_node,j_node = np.min(nodes),np.max(nodes)
				edge = (i_node,j_node)
				# print (edge)
				edges.append(edge)
				years.append(year)
	edges,year_idx,weight = np.unique(edges,axis=0,return_index=True,return_counts=True) 
	years = np.array(years)[np.array(year_idx)]   
	g.add_edges(edges) 
	g.es['year'] = years
				
	g.write_pickle('/%s/data/%s_coa_graph'%(homedir,method))
	np.save('/%s/data/%s_a_2_node.npy'%(homedir,method), a_2_node) 
	np.save('/%s/data/%s_a_2_paper.npy'%(homedir,method), a_2_paper) 
	np.save('/%s/data/%s_node_2_a.npy'%(homedir,method), node_2_a) 

def write_graph():
	# for year in np.unique(author_df.PY.values):
	g = igraph.load('/%s/data/%s_coa_graph'%(homedir,method),format='pickle')  
	g.es.select(year_ne=2019).delete()
	g = g.clusters().giant()
	vc = g.community_fastgreedy().as_clustering(4)


	membership = vc.membership
	from matplotlib.colors import rgb2hex
	color_array = np.zeros((g.vcount())).astype(str)
	colors = np.array([[72,61,139],[82,139,139],[180,205,205],[205,129,98]])/255.
	for i in range(g.vcount()):
		color_array[i] = rgb2hex(colors[membership[i]])
	g.vs['color'] = color_array.astype('str')
	g.vs['sp'] = g.shortest_paths(18)[0]

	max_path = np.max(g.vs['sp'][0])	

	def walk(i):
		walked = np.zeros(g.vcount())
		walk_2 = 18
		for i in range(max_path*2):
			walk_2 = np.random.choice(g.neighbors(walk_2))
			walked[walk_2] += 1
		return walked
	pool = multiprocessing.Pool(8)
	walked = pool.map(walk,range(100000))

	sum_walk = np.sum(walked,axis=0)
	final_walk = np.zeros((g.vcount()))

	final_walk[sum_walk>0] = np.argsort(sum_walk[sum_walk>0]) 
	# walked[100] = 0

	g.vs['walks'] = final_walk
	g.write_gml('/%s/citation_network_%s_big.gml'%(homedir,method))

	from igraph import VertexClustering
	race_vs = VertexClustering.FromAttribute(g,'race')                                                                                                 

	print (vcc.modularity)
	print (race_vs.modularity)

def analyze_coa_mod():
	r = pd.read_csv('/%s/data/result_df_%s_all.csv'%(homedir,method))
	r.name = r.name.str.strip()
	node_2_a = np.load('/%s/data/%s_node_2_a.npy'%(homedir,method),allow_pickle='TRUE').item()
	g = igraph.load('/%s/data/%s_coa_graph'%(homedir,method),format='pickle')

	race_prs = np.zeros((g.vcount(),4))
	for node in tqdm.tqdm(range(g.vcount()),total=g.vcount()):
		asian, black, hispanic, white = r[r.name==node_2_a[node].strip()].values[0][4:]
		race_prs[node] = [white,asian,hispanic,black]

	race_prs = race_prs / np.sum(race_prs,axis=1)[:,None]  
	g.race_prs = race_prs

	def mod(year):
		yg = g.copy()
		yg.es.select(year_gt=year).delete()
		race_binary = np.zeros((g.vcount()))
		for node in range(g.vcount()):
			race_binary[node] = np.random.choice([1,2,3,4],p=g.race_prs[node])
		yg.vs['race'] = race_binary
		yg = yg.clusters().giant()

		rm = VertexClustering.FromAttribute(yg,'race').modularity
		em = yg.community_infomap().modularity 
		return ([year,rm,em])

	pool = multiprocessing.Pool(25)
	years = []
	for i in range(100):
		for y in np.arange(1995,2020):
			years.append(y)
	q_vals =  pool.map(mod,years)

	years = []
	races = []
	emperical = []
	for q in q_vals:
		years.append(q[0])
		races.append(q[1])
		emperical.append(q[2])


	np.save('/%s/data/q_analysis_emp_%s.npy'%(homedir,method),emperical)
	np.save('/%s/data/q_analysis_year_%s.npy'%(homedir,method),years)
	np.save('/%s/data/q_analysis_race_%s.npy'%(homedir,method),races)

def plot_coa_mod():
	emperical = np.load('/%s/data/q_analysis_emp_%s.npy'%(homedir,method))
	years = np.load('/%s/data/q_analysis_year_%s.npy'%(homedir,method))
	races = np.load('/%s/data/q_analysis_race_%s.npy'%(homedir,method))

	race_df = pd.DataFrame(np.array([races,years]).transpose(),columns=['q','year'])     
	race_df['network partition'] = 'race'
	e_df = pd.DataFrame(np.array([emperical,years]).transpose(),columns=['q','year'])     
	e_df['network partition'] = 'q-max'
	df = race_df.append(e_df,ignore_index=True)

	sns.set(style='white',font='Palatino')
	plt.close()
	fig = plt.figure(figsize=(3,2.5),constrained_layout=True)
	# y1 = sns.lineplot(np.unique(g.es['year']),(race-emperical)/emperical)  
	y1=sns.lineplot(years,races,color='blue',label='race',legend=False,ci='sd',n_boot=1000)
	

	plt.ylabel('Q, race partition',color='blue')
	y2 = y1.axes.twinx()
	plt.sca(y2)
	# sns.lineplot(np.unique(g.es['year']),race)  
	sns.lineplot(years,emperical,color='black',label='emperical',ci='sd',legend=False,n_boot=1000)
	plt.ylabel('Q, emperical partition')
	# y2.legend(loc=7)
	# y1.figure.legend(loc=3)
	y1.set_xticks([1995,2003,2011,2020]) 
	plt.savefig('q_analysis.pdf')

def multi_pr_shortv2(paper):
	global paper_df
	global main_df
	global graphs
	global node_2_a
	global a_2_node
	global a_2_paper
	global walk_length
	global prs
	print (paper)
	this_paper_main_df = main_df.iloc[paper]
	citing_authors = [this_paper_main_df.AF.split(';')[0].strip(),this_paper_main_df.AF.split(';')[-1].strip()]
	year = this_paper_main_df.PY
	yg,m,big = graphs[year]
	# get authors who are cited
	cited = this_paper_main_df.cited.split(', ')
	cited_authors = []
	for c in cited:
		cited_df = main_df.iloc[int(c)-1]
		for ca in cited_df.AF.split('; '):
			cited_authors.append(ca.strip())


	cited_authors = np.unique(cited_authors).flatten()
	cited_authors = np.setdiff1d(cited_authors,citing_authors) 
	# get shortest paths to papers cited
	all_paths = []
	cited_paths= []
	for i in citing_authors:
		i = a_2_node[i]
		all_paths.append(yg.shortest_paths(i))
		js = []
		for j in cited_authors: 
			j = a_2_node[j]
			js.append(j)
		cited_paths.append(yg.shortest_paths(i,js))
	cited_paths = np.array(cited_paths).reshape(1,-1)
	cited_paths[np.isinf(cited_paths)] = -1
	cited_paths[cited_paths==-1] = np.nanmax(cited_paths) + 1
	assert np.min(cited_paths) >= 1

	all_paths = np.array(all_paths).reshape(2,-1)
	all_paths[np.isinf(all_paths)] = -1
	all_paths[all_paths==-1] = np.nanmax(all_paths) + 1

	null_walks = 0

	base_matrix = np.zeros((8,8))
	base_matrix[:] = 0

	#if we take random walks, the length of the longest random path between two nodes, what is the base rate?
	# we only count times when the walk ends on a first or last author and paper was published before this paper
	
	if walk_length[:3] == 'all':
		try:
			multi = int(walk_length[4])
			wl = int(np.nanmax(all_paths))*multi
		except: wl = int(np.nanmax(all_paths))
		if wl == 0:
			base_matrix[:] = np.nan
			return [paper,base_matrix,np.mean(cited_paths),np.mean(all_paths)]
		for i in citing_authors:
			i = a_2_node[i.strip()]
			while True:
				walk_2 = i # start from this author
				for w in range(wl):
					walk_2 = np.random.choice(yg.neighbors(walk_2),1)[0]
				if walk_2 == i: continue
				cited_author = node_2_a[walk_2]
				walk_papers = a_2_paper[cited_author]
				np.random.shuffle(walk_papers)
				for walk_paper in walk_papers:
					walk_paper_df = main_df.iloc[walk_paper]
					if walk_paper_df.PY>= year: continue
					#you found a paper, store it in matrix! 
					base_matrix = base_matrix + prs[walk_paper]
					null_walks += 1
				if null_walks > 1000:break
	if walk_length == 'cited':
		null_walks = 0.
		while True:
			wl = np.random.choice(cited_paths[0],1)[0]#pick a length from citations
			choices = np.where(all_paths[0]==wl)[0]
			if len(choices) == 0: continue
			walk_paper = np.random.choice(choices,1)[0] #pick a cited author of that length, any author
			cited_author = node_2_a[walk_paper]
			walk_papers = a_2_paper[cited_author]
			np.random.shuffle(walk_papers)
			for walk_paper in walk_papers:
				walk_paper_df = main_df.iloc[walk_paper]
				if walk_paper_df.PY>= year: continue
				#you found a paper, store it in matrix! 
				base_matrix = base_matrix + prs[walk_paper]
				null_walks += 1
			if null_walks > 1000:break
		null_walks = 0.
		while True:
			wl = np.random.choice(cited_paths[0],1)[0]#pick a length from citations
			choices = np.where(all_paths[1]==wl)[0]
			if len(choices) == 0: continue
			walk_paper = np.random.choice(choices,1)[0] #pick a cited author of that length, last author
			cited_author = node_2_a[walk_paper]
			walk_papers = a_2_paper[cited_author]
			np.random.shuffle(walk_papers)
			for walk_paper in walk_papers:
				walk_paper_df = main_df.iloc[walk_paper]
				if walk_paper_df.PY>= year: continue
				#you found a paper, store it in matrix! 
				base_matrix = base_matrix + prs[walk_paper]
				null_walks += 1
			if null_walks > 1000:break

	return [paper,base_matrix,np.mean(cited_paths),np.mean(all_paths)]

def make_year_graphs(year):
	global g
	print (year)
	yg = g.copy()
	yg.es.select(year_gt=year).delete()
	m = np.array(yg.components().membership)
	labels,counts = np.unique(m,return_counts=True)
	big = np.argmax(counts)
	return [year,[yg,m,big]]

def shortest_pr_paths():
	global paper_df
	global main_df
	global graphs
	global node_2_a
	global a_2_node
	global a_2_paper
	global g
	global walk_length
	global prs 
	prs = np.load('/%s/data/result_pr_df_%s.npy'%(homedir,method))
	g = igraph.load('/%s/data/%s_coa_graph'%(homedir,'wiki'),format='pickle')
	main_df = pd.read_csv('/%s/article_data/NewArticleData2019_filtered.csv'%(homedir),header=0)


	a_2_node = np.load('/%s/data/%s_a_2_node.npy'%(homedir,method),allow_pickle='TRUE').item()
	a_2_paper = np.load('/%s/data/%s_a_2_paper.npy'%(homedir,method),allow_pickle='TRUE').item()
	node_2_a = np.load('/%s/data/%s_node_2_a.npy'%(homedir,method),allow_pickle='TRUE').item()

	pool = multiprocessing.Pool(multiprocessing.cpu_count())
	graphs = dict(pool.map(make_year_graphs,np.arange(2009,2020)))

	del pool
	paper_idxs = []
	for i,p in main_df.iterrows():
		year = p.PY
		if year >=2009:
			if type(p.cited) != np.float:
				if len(p['cited'].split(',')) >= 10:
					if graphs[year][1][a_2_node[p.AF.split(';')[0].strip()]] == graphs[year][2]:
						if graphs[year][1][a_2_node[p.AF.split(';')[-1].strip()]] == graphs[year][2]:
							paper_idxs.append(i)

	pool = multiprocessing.Pool(multiprocessing.cpu_count())
	r = pool.map(multi_pr_shortv2,paper_idxs[::-1])


	walk_prs = np.zeros((main_df.shape[0],64))
	walk_prs[:] = np.nan
	walk_all = np.zeros((main_df.shape[0]))
	walk_all[:] = np.nan
	walk_cite = np.zeros((main_df.shape[0]))
	walk_cite[:] = np.nan
	for result in r:
		walk_prs[result[0]] = result[1].flatten()
		walk_cite[result[0]] = result[2]
		walk_all[result[0]] = result[3]

	np.save('/%s/data/walk_pr_probabilities_%s_%s.npy'%(homedir,method,walk_length),walk_prs)	
	np.save('/%s/data/walk_pr_all_%s_%s.npy'%(homedir,method,walk_length),walk_all)	
	np.save('/%s/data/walk_pr_cite_%s_%s.npy'%(homedir,method,walk_length),walk_cite)	

def cite_paths():
	control = False
	df = pd.read_csv('/%s/article_data/NewArticleData2019_filtered.csv'%(homedir),header=0)

	df = pd.read_csv('/%s/data/result_df_%s.csv'%(homedir,method))


	# walk_all = np.load('/%s/data/walk_all_%s_%s.npy'%(homedir,method,walk_length))	
	walk_cite = np.load('/%s/data/walk_pr_cite_%s_%s.npy'%(homedir,method,walk_length))	

	mask = np.isnan(walk_cite)==False
	# print (scipy.stats.ttest_rel(walk_all[mask],walk_cite[mask]))

	matrix = np.load('/%s/data/citation_matrix_pr_%s_%s.npy'%(homedir,method,control))
	base_matrix = np.load('/%s/data/base_citation_matrix_pr_%s_%s.npy'%(homedir,method,control))




	matrix_idxs = {'white_M':0,'api_M':1,'hispanic_M':2,'black_M':3,'white_W':4,'api_W':5,'hispanic_W':6,'black_W':7}

	if within_poc == False:
		groups = [np.vectorize(matrix_idxs.get)(['white_M','white_W',]),
		np.vectorize(matrix_idxs.get)(['api_M','api_W','hispanic_M','hispanic_W','black_M','black_W',])]
		names = ['white-white','white-poc','poc-white','poc-poc']

	if within_poc == 'black':
		groups = [np.vectorize(matrix_idxs.get)(['white_M','white_W',]),
		np.vectorize(matrix_idxs.get)(['black_M','black_W',])]
		names = ['nb-nb','nb-black','black-nb','black-black']

	if within_poc == 'api':
		groups = [np.vectorize(matrix_idxs.get)(['white_M','white_W',]),
		np.vectorize(matrix_idxs.get)(['api_M','api_W',])]
		names = ['na-na','na-asian','asian-na','asian-asian']

	if within_poc == 'hispanic':
		groups = [np.vectorize(matrix_idxs.get)(['white_M','white_W',]),
		np.vectorize(matrix_idxs.get)(['hispanic_M','hispanic_W',])]
		names = ['nh-nh','nh-hispanic','hispanic-nh','hispanic-hispanic']


	plot_matrix = np.zeros((matrix.shape[0],len(groups),len(groups)))
	plot_base_matrix = np.zeros((matrix.shape[0],len(groups),len(groups)))

	for i in range(len(groups)):
		for j in range(len(groups)):
			plot_matrix[:,i,j] = np.nansum(matrix[:,groups[i]][:,:,groups[j]].reshape(matrix.shape[0],-1),axis=1)
			plot_base_matrix[:,i,j] = np.nansum(base_matrix[:,groups[i]][:,:,groups[j]].reshape(matrix.shape[0],-1),axis=1)


	jdf = pd.DataFrame()
	jdf['fa_race'] = df.fa_race
	jdf['la_race'] = df.la_race
	jdf['expected_ww'] = plot_base_matrix[:,0,0]
	jdf['observed_ww'] = plot_matrix[:,0,0]
	jdf['expected_aa'] = plot_base_matrix.reshape(df.shape[0],4)[:,1:].sum(axis=1)
	jdf['observed_aa'] = plot_matrix.reshape(df.shape[0],4)[:,1:].sum(axis=1)
	jdf['path2cited'] = walk_cite

	jdf[mask].to_csv('cited_paths_%s.csv'%(method))
	1/0

	n_iters = 100

	white_data = pd.DataFrame(columns=['citation_rate','citation_type','path_length'])
	papers = df[(df.year>2009)&((df.fa_race=='white')&(df.la_race=='white'))]
	for boot in range(n_iters):
		boot_papers = papers.sample(100,replace=False).index
		
		emperical = np.nanmean(plot_matrix[boot_papers],axis=0)
		expected = np.nanmean(plot_base_matrix[boot_papers],axis=0)
		
		emperical = emperical / np.sum(emperical)
		expected = expected / np.sum(expected)

		rate = (emperical - expected) / expected
		
		pl = np.nanmean(walk_cite[boot_papers])
		# np.sum(rate.flatten()[1:])-rate[0,0]
		
		tdf = pd.DataFrame(data=np.array([rate.flatten(),names]).swapaxes(0,1),columns=['citation_rate','citation_type'])
		tdf['path_length'] = pl
		white_data = white_data.append(tdf,ignore_index=True)   
	
	white_data.citation_rate = (white_data.citation_rate.astype(float)*100) #are you citing aoc well?

		
	for perc in [10,15,20,25]:

		x = []
		y = []
		for name in names[3:]:
			cut_off = np.percentile(white_data[(white_data.citation_type==name)].citation_rate,perc)
			x.append(white_data[(white_data.citation_type==name)&(white_data.citation_rate>cut_off)].path_length.values)
			y.append(white_data[(white_data.citation_type==name)&(white_data.citation_rate<cut_off)].path_length.values)
		x = np.array(x).flatten()
		y = np.array(y).flatten()
		print (scipy.stats.ttest_ind(x,y))
		
	for perc in [10,15,20,25]:
		x = []
		y = []
		cut_off = np.percentile(white_data[(white_data.citation_type=='white-white')].citation_rate,perc)
		x.append(white_data[(white_data.citation_type=='white-white')&(white_data.citation_rate>cut_off)].path_length.values)
		y.append(white_data[(white_data.citation_type=='white-white')&(white_data.citation_rate<cut_off)].path_length.values)
		x = np.array(x).flatten()
		y = np.array(y).flatten()
		print (scipy.stats.ttest_ind(x,y))

	data = pd.DataFrame(columns=['citation_rate','path_length'])
	papers = df[(df.year>2009)&((df.fa_race!='white')|(df.la_race!='white'))]
	for boot in range(n_iters):
		boot_papers = papers.sample(100,replace=False).index
		
		emperical = np.nanmean(plot_matrix[boot_papers],axis=0)
		expected = np.nanmean(plot_base_matrix[boot_papers],axis=0)
		
		emperical = emperical / np.sum(emperical)
		expected = expected / np.sum(expected)

		rate = (emperical - expected) / expected
		
		pl = np.nanmean(walk_cite[boot_papers])

		tdf = pd.DataFrame(data=np.array([rate.flatten(),names]).swapaxes(0,1),columns=['citation_rate','citation_type'])
		tdf['path_length'] = pl
		data = data.append(tdf,ignore_index=True)  

	data.citation_rate = (data.citation_rate.astype(float)*100) #are you citing aoc well?

	for perc in [10,15,20,25]:


		x = []
		y= []
		for name in names[1:]:
			cut_off = np.percentile(data[(data.citation_type==name)].citation_rate,perc)
			x.append(data[(data.citation_type==name)&(data.citation_rate>cut_off)].path_length.values)
			y.append(data[(data.citation_type==name)&(data.citation_rate<cut_off)].path_length.values)
		x = np.array(x).flatten()
		y = np.array(y).flatten()
		print (scipy.stats.ttest_ind(x,y))


	for perc in [10,15,20,25]:
		x = []
		y= []
		cut_off = np.percentile(data[(data.citation_type=='white-white')].citation_rate,perc)
		x.append(data[(data.citation_type=='white-white')&(data.citation_rate>cut_off)].path_length.values)
		y.append(data[(data.citation_type=='white-white')&(data.citation_rate<cut_off)].path_length.values)
		x = np.array(x).flatten()
		y = np.array(y).flatten()
		print (scipy.stats.ttest_ind(x,y))

def make_all_rates():
	
	pool = multiprocessing.Pool(4)
	pool.map(make_pr_percentages,[True,'null_True',False,'null_False'])
	del pool


	walk_length = 'all'
	pool = multiprocessing.Pool(2)
	pool.map(make_pr_percentages,['walk','null_walk'])
	del pool
	
	walk_length = 'cited'
	pool = multiprocessing.Pool(2)
	pool.map(make_pr_percentages,['walk','null_walk'])
	del pool

# 1/0
# print (control)
# make_df()
# make_pr_df()
# make_all_author_race()
# if control == True: make_pr_control()
# if control == 'True_jn': make_pr_control_jn()
# if control == 'walk': 
# 	# make_networks()
# 	shortest_pr_paths()
# make_pr_percentages(control)
# make_pr_percentages('null_%s'%(control))
# plot_pr_percentages_booty_matrix([control,False,False])
# plot_pr_intersections(control,'all')
# for c in ['aa','aw','wa','ww']:
# 	plot_pr_intersections(control,c)





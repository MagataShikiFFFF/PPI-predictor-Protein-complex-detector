#################################complexes data from CYC2008 dataset##########################
#import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

pc_df = pd.read_excel('CYC2008_complex.xls') 


orf = pd.read_fwf('yeast.txt') 
df_orf = orf.drop(columns=['Gene designations', 'Unnamed: 1', 'SGD', 'Size 3D', 'CH','Unnamed: 7', 'Unnamed: 8'])  

def choose(x, y, orf_data):
	d_seq = {}
	seq = list(SeqIO.parse(x, "fasta")) 
	pc_data = pd.read_excel(y)

	for i in range(len(pc_data)):
		for j in range(len(orf_data)):
			if orf_data.at[j, 'OLN']==pc_data.at[i, 'ORF']:
				if orf_data.at[j, 'OLN'] not in d_seq.keys():
					d_seq[orf_data.at[j, 'OLN']] = []

				for k in range(len(seq)):
					if orf_data.at[j, 'SP'] == seq[k].id.split('|')[1]:
						d_seq[orf_data.at[j, 'OLN']].append(seq[k].seq)

	for n in d_seq.keys():
		if d_seq[n]==[]:
			d_seq = site.pop[n]  

	return d_seq


pop_obj=site.pop('name')


seq_data = choose("uniprot-(yeast).fasta ", "CYC2008_complex.xls", df_orf)

def choose(x, y, orf_data):
	d_seq = {}
	seq = list(SeqIO.parse(x, "fasta")) 
	pc_data = pd.read_excel(y)

	for i in range(len(pc_data)):
		for j in range(len(orf_data)):
			for k in range(len(seq)):
				if orf_data.at[j, 'OLN']==pc_data.at[i, 'ORF']:
					if orf_data.at[j, 'SP'] == seq[k].id:
						if orf_data.at[j, 'OLN'] not in d_seq.keys():
							d_seq[orf_data.at[j, 'OLN']] = []
						d_seq[orf_data.at[j, 'OLN']].append(seq[k].seq)

	return d_seq

def cluster_lab(pc_df, orf_data):
    
	pc_data = pd.read_excel(pc_df)

	dic_pc= {}

	for i in range(len(pc_data)):
		if pc_data.at[i, 'Complex'] not in dic_pc.keys():
			dic_pc[pc_data.at[i, 'Complex']] = []
		#dic_pc[pc_data.at[i, 'Complex']].append(pc_data.at[i, 'ORF'])

		for j in range(len(orf_data)):
			if orf_data.at[j, 'OLN']==pc_data.at[i, 'ORF']:
				sp = orf_data.at[j, 'SP']
				dic_pc[pc_data.at[i, 'Complex']].append(sp)

	return dic_pc

cl_label = cluster_lab('CYC2008_complex.xls', df_orf)

def cluster_lab_orf(pc_df):
    
	pc_data = pd.read_excel(pc_df)

	dic_pc= {}

	for i in range(len(pc_data)):
		if pc_data.at[i, 'Complex'] not in dic_pc.keys():
			dic_pc[pc_data.at[i, 'Complex']] = []
		dic_pc[pc_data.at[i, 'Complex']].append(pc_data.at[i, 'ORF'])

	return dic_pc

cl_label = cluster_lab_orf('CYC2008_complex.xls')

cl_lab = dict(zip(range(len(cl_label)), cl_label.values()))

num = []
lab = []
for i in cl_lab.keys():  
	if len(cl_lab[i])>15:  
		num.append(i)
		for j in cl_lab[i]:
			if j not in lab:
				lab.append(j)


def orf_ppi(pc_df, ppi_df, orf_data):
	#pc_data = pd.read_excel(pc_df)
	dic_pc = {}
	dic_data = []
	for i in pc_df:
		for j in range(len(orf_data)):
			if i == orf_data.at[j, 'OLN']:
				sp = orf_data.at[j, 'SP']
				for k in ppi_df.keys():
					if sp in ppi_df[k]:	
						if k not in dic_pc.keys():
							dic_pc[k] = []
						if i not in dic_pc[k]:
							dic_pc[k].append(i)
						if i not in dic_data:
							dic_data.append(i)

	return dic_pc, dic_data

orf_ppni, orf_n_node = orf_ppi('CYC2008_complex.xls', id_DIP_p, df_orf)

orf_ppi_n, orf_node_n = orf_ppi(lab_new, id_DIP_p, df_orf)

orf_label = {}

for i in orf_node: 
	for j in range(len(pc_df)): 
		if i==pc_df.at[j, 'ORF']: 
			if pc_df.at[j, 'Complex'] not in orf_label.keys(): 
				orf_label[pc_df.at[j, 'Complex']] = [] 
            orf_label[pc_df.at[j, 'Complex']].append(i)

orf_label = dict(zip(range(len(orf_label)), orf_label.values())) 

new_ppi = {} 
for i in orf_ppi.keys(): 
    if len(orf_ppi[i]) == 2: 
        if i not in new_ppi.keys(): 
            new_ppi[i] = [] 
        new_ppi[i].append(orf_ppi[i][0]) 
        new_ppi[i].append(orf_ppi[i][1])  


def mk_label(dic_label, p_node):
	orf_lab = {}
	for i in p_node: 
		for j in dic_label.keys(): 
			if i in dic_label[j]:
				if i not in orf_lab.keys():
					orf_lab[i] = []
				orf_lab[i].append(j)


	
	return orf_lab
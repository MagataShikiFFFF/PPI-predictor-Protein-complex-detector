import numpy as np
import pandas as pd
import time
import h5py
import pickle 

from nltk.util import ngrams
from Bio import SeqIO
import os
import itertools

def reader_data(x):
	num = [] 
	num2 = {} 
	with open(x, 'r') as f: 
		for line in f.readlines(): 
			num.append(line.split('\t'))

		for i in range(len(num)): 
			if num[i][0] not in num2.keys(): 
				num2[num[i][0]] = [] 
			for j in num[i][5].split(';'): 
				if j not in num2[num[i][0]]: 
					num2[num[i][0]].append(j)

	return num2 

mips = reader_data("allComplexes.txt") 

def part1(cl_, num):
	cl_label_ = {}
	for i in cl_.keys():
		if len(cl_[i])>num:
			if i not in cl_label_.keys():
				cl_label_[i] = []
				for j in cl_[i]:
					cl_label_[i].append(j)

	return cl_label_

mips_d1 = dict(zip(range(len(part1(mips, 40))), part1(mips, 40).values()))

lab = []
for i in mips_d1.keys():
	for j in mips_d1[i]:
		if j not in lab:
			lab.append(j)

def choose(x, pc_data):
	d_seq = {}
	seq = list(SeqIO.parse(x, "fasta")) 
	#pc_data = pd.read_excel(y)

	for i in pc_data:
		if i not in d_seq.keys():
			d_seq[i] = []
		for k in range(len(seq)):
			if i == seq[k].id.split('|')[1]:
				d_seq[i].append(seq[k].seq)

	for n in list(d_seq):
		if d_seq[n]==[]:
			d_seq.pop(n, None)
			#d_seq = ites.pop[n] 

	return d_seq

seq_data1 = choose("uniprot-(Human).fasta", lab)
seq_data2 = choose("uniprot-(Mouse).fasta", lab)
seq_data3 = choose("uniprot-(Rat).fasta", lab)
seq_all = dict(seq_data1, **seq_data2)
seq_all = dict(seq_all, **seq_data3)

lab = list(seq_all.keys())

new = []
for i in range(len(list(seq_all.keys()))):
	tmp = list((list(seq_all.keys())[i], list(seq_all.values())[i][0]))
	new.append(tmp)

with open("semi_mips_seq_id.txt", "rb") as fp: 
    new = pickle.load(fp)

from itertools import combinations

pair = list(combinations(new, 2))

np.random.shuffle(pair)

def mk_label(dic_label, p_node):
	orf_lab = {}
	lab_data = []
	for i in p_node: 
		for j in dic_label.keys(): 
			if i in dic_label[j]:
				if i not in orf_lab.keys():
					orf_lab[i] = []
				orf_lab[i].append(j) 

	for k in orf_lab.keys():
		lab_data.append(orf_lab[k][0])
	lab_data = np.array(lab_data) 
	
	return orf_lab, lab_data

y_dic, y_data = mk_label(mips_d1, lab)



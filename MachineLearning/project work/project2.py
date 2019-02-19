# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 11:17:36 2018

@author: mingyux
"""
#%%
#project 2
import time
import os
from os import listdir
from os.path import isfile, join
import featext.feature_processing
from featext.mfcc import Mfcc
import numpy as np
import system.gmm_em as gmm
import system.ivector as ivector
import system.backend as backend
from sklearn.metrics import roc_curve
import linecache
## counting time
start = time.clock()

N_COMPONENTS = 1024
IVECTOR_DIM = 400
PLDA_DIM = 200
nworkers = 8
EXTRACT_FEATURES = 0
FEATURE_FOLDER ='G:\\UEF-lecture\\SummerSchool\\project2\\celeb_match_project\\feature_folder\\'
STATISTICS_FOLDER ='G:\\UEF-lecture\\SummerSchool\\project2\\celeb_match_project\\statistics_folder\\'
print('1')



# UPDATE THIS FOLDER (folder to spoken digit dataset recordings):
#data_folder = '/home/ville/files/recordings/'
#data_folder = 'G:\\UEF-lecture\\SummerSchool\\clips\\'

#speakers = ['jackson', 'nicolas', 'theo']
#n_speakers = len(speakers)
#n_speakers = 1 # there is one speaker 
#n_digits = 10   # 0 - 9
#n_digits = 1   # 0 - 9
#n_sessions = 50   # 0 - 49
#n_sessions = 5   # 0 - 49

#### Feature extraction:
# MFCCs

mfcc = Mfcc()
mfcc.fs = 16000
mfcc.max_frequency = 7000
mfcc.min_frequency = 200
mfcc.frame_duration = 0.025
mfcc.frame_overlap_duration = 0.01
mfcc.sad_threshold = 40
mfcc.include_deltas = 1
mfcc.include_double_deltas = 1
mfcc.include_base_coeffs = 1
mfcc.include_energy = 1
mfcc.n_coeffs = 20
mfcc.rasta_coeff = 0
mfcc.pre_emphasis = 0
mfcc.cmvn = 1
mfcc.initialize()




features= featext.feature_processing.extract_features_from_file(
       # filelist_path='voxceleb_lists/voxceleb_files_quarter.txt',
        file_path='G:\\UEF-lecture\\SummerSchool\\clips\\1.wav',
        feature_extractor=mfcc,
        channel=0)
id = '1'
filename = id + '.npy'
output_file = os.path.join(FEATURE_FOLDER, filename)
np.save(output_file,features);
print('2')

#### Sufficient statistics (Baum-Welch statistics) extraction:
ubm = gmm.GMM(ndim=60, nmix=N_COMPONENTS, ds_factor=1, final_niter=10, nworkers=nworkers);
print('ubm1');
ubm.load('G:\\UEF-lecture\\SummerSchool\\project2\\celeb_match_project\\ubm.npz');
print('ubm2')
featext.feature_processing.extract_all_stats(feature_folder=FEATURE_FOLDER,ubm=ubm,output_folder=STATISTICS_FOLDER,nworkers=nworkers);
print('ubm')

#### I-vector extraction:
tMatrix = ivector.TMatrix(IVECTOR_DIM, 60, N_COMPONENTS, niter=5, nworkers=nworkers)
tMatrix.Tm = np.load('tmatrix.npy')

extractor = ivector.Ivector(IVECTOR_DIM, 60, N_COMPONENTS)
extractor.initialize(ubm, tMatrix.Tm)
stat_files = [f for f in os.listdir(STATISTICS_FOLDER) if os.path.isfile(os.path.join(STATISTICS_FOLDER, f))]
ivec_map = {}
for idx, filename in enumerate(stat_files):
    id = filename[:-4]
    statdata = np.load(os.path.join(STATISTICS_FOLDER, filename))
    ivec_map[id] = extractor.extract(statdata['n'], statdata['f'])
    if idx % 1000 == 0:
        print('I-vector extraction {}/{}'.format(idx, len(stat_files)))
print('3')
 
mw = np.load('mean_ivector_and_whitening_matrix.npz')

#### I-vector processing
CM = mw['c']
WM = mw['w']
print('Separating training vectors from the rest')
my_vectors = np.zeros((IVECTOR_DIM, 1))
for idx, id in enumerate('1'):
    my_vectors[:, idx] = ivec_map[id]

#### I-vector processing
print('I-vector processing')
#center = backend.compute_mean(training_vectors)
#w = backend.calc_white_mat(np.cov(training_vectors))
#np.savez(MW_PATH, c=center, w=w)
my_vectors = backend.preprocess(my_vectors, CM, WM)
print('4')
 
#### PLDA  

plda = backend.GPLDA(IVECTOR_DIM, PLDA_DIM, niter=20)
plda_load = np.load('plda.npz')
plda.St = plda_load['St']
plda.Sb = plda_load['Sb']
#### celebrities vectors

cele_vectors = np.load('ivectors.npy')
#### Scoring:
scores = plda.score_trials(cele_vectors, my_vectors)
print('5')
 
#### load speaker file
speaker_ids=[]
with open('speaker_ids.txt', 'r') as f:
    data = f.readlines()  
 
    for line in data:
        id_speaker =  line[2:7]  #get id and store into numpy array
        speaker_ids.append( id_speaker)
#### load sentences file
sentence_ids=[]       
with open('sentence_ids.txt', 'r') as f:
    data_sentences = f.readlines()  
 
    for line_se in data_sentences:
        
        sentence_ids.append(line_se)        
#### calculate the average: get scores for every celebrities
unique_speakers =  list(set(speaker_ids))# calculate how many unique speakers
avg_speakers=[] # store the averaging scores
for speaker in unique_speakers:
    
    count_score = 0 #counting score
    count_duplicate = 0 # how many duplicate
    index = 0# location for the speaker
    for speaker_temp in speaker_ids:
        if  speaker==speaker_temp :
            count_score = count_score + scores[index]    
            count_duplicate = count_duplicate + 1
        index = index + 1
    avg_temp = count_score / count_duplicate # average score for one speaker
    avg_speakers.append(avg_temp)   
print('6')     
 
#### Find the closest speaker's and segments from Youtube
highest_score_speaker = np.argmax(avg_speakers, axis=0)
highest_speaker_id = unique_speakers[int(highest_score_speaker)]


 

#### find the corresponding segment
Highest_ce_list=[]
for i,x in enumerate(speaker_ids):
    if x==highest_speaker_id:
       Highest_ce_list.append(i)
Highest_ce_scores = scores[Highest_ce_list]
highest_score_segments = np.argmax(Highest_ce_scores, axis=0)
highest_score_segments_index = Highest_ce_list[int(highest_score_segments)]
Expected_segment = sentence_ids[highest_score_segments_index ]
#### splicing the string to an url
part = Expected_segment.split('_')

print('7')
 
#### get the corresponding segments from timestamp
TimeStamp_path = 'G:\\UEF-lecture\\SummerSchool\\dev\\txt\\id' + highest_speaker_id +'\\'+ part[0] + '\\'+part[1][2:7]+'.txt'
start_Timestamp = linecache.getline(TimeStamp_path,8)
end_Timestamp = open(TimeStamp_path).readlines()[-1]
start_Time_split = start_Timestamp.split(' ') 
end_Time_split = end_Timestamp.split(' ')
#### convert to seconds
start_second = int(start_Time_split[0])/25
end_second = int(end_Time_split[0])/25
url ='https://youtu.be/'+ part[0]+'?t='+ str(int(start_second))
#### convert to mm:ss
start_mm = int(start_second/60)
start_ss = start_second - start_mm * 60
end_mm = int(end_second/60)
end_ss = end_second - end_mm * 60
print('8')
#### output the information
print('1.the id of closest speaker is:'+highest_speaker_id+'\n'+'2.the url is:'+url
      +'\n3.the cloeset segment is from '+str(start_mm)+' minute '+str(start_ss)+' seconds to '
      +str(end_mm)+' minute '+str(end_ss)+' seconds\n')
## counting time cost
elapsed = (time.clock() - start)
print('Time used:'+str(elapsed)+' seconds') 
 
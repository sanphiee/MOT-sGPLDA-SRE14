%{ 

This is a demo on how to use the Identity Toolbox for i-vector based speaker
recognition. A relatively small scale task has been designed using speech
material from the TIMIT corpus. There are a total of 630 (192 female and
438 male) speakers in TIMIT, from which we have selected 530 speakers for
background model training and the remaining 100 (30 female and 70 male)
speakers are used for tests. There are 10 short sentences per speaker in
TIMIT. For background model training we use all sentences from all 530
speakers (i.e., 5300 speech recordings in total). For speaker specific
model training we use 9 out of 10 sentences per speaker and keep the
remaining 1 sentence for tests. Verification trials consist of all possible
model-test combinations, making a total of 10,000 trials (100 target vs
9900 impostor trials).

Assuming that audio recordings are already converted 
into cepstral features, there are 5 steps involved:
 
 1. training a UBM from background data
 2. learning a total variability subspace from background statistics
 3. training a Gaussian PLDA model with development i-vectors
 4. scoring verification trials with model and test i-vectors
 5. computing the performance measures (e.g., EER)

Note: given the relatively small size of the task, we can load all the data 
and models into memory. This, however, may not be practical for large scale 
tasks (or on machines with a limited memory). In such cases, the parameters 
should be saved to the disk.

Omid Sadjadi <s.omid.sadjadi@gmail.com>
Microsoft Research, Conversational Systems Research Center

%}

clc;
clear all;
close all;

load '../temp/sre14.mat';

% for nphi=50:50:250
    
nphi = 250; % dimension
niter = 50; % iteration

%% train plda
plda = gplda_em(dev_ivec', dev_label, nphi, niter);

%% score plda
PLDA_scores = score_gplda_trials(plda, enrol_ivec', test_ivec');
PLDA_scores_col = reshape(PLDA_scores',length(PLDA_scores(:)),1);

figure;
score_pos = find(test_mask < 2.5);
all_scores = PLDA_scores_col(score_pos);
all_key = test_key(score_pos);
[PLDA_eer,PLDA_dcf08,PLDA_dcf10,PLDA_dcf14] = compute_eer(all_scores,all_key,true);

figure;
prog_pos = find(test_mask==1);
prog_score = PLDA_scores_col(prog_pos);
prog_key = test_key(prog_pos);
[prog_PLDA_eer, prog_PLDA_dcf08, prog_PLDA_dcf10, prog_PLDA_dcf14] = compute_eer(prog_score,prog_key,true);

figure;
eval_pos = find(test_mask==2);
eval_score = PLDA_scores_col(eval_pos);
eval_key = test_key(eval_pos);
[eval_PLDA_eer, eval_PLDA_dcf08, eval_PLDA_dcf10, eval_PLDA_dcf14] = compute_eer(eval_score,eval_key,true);

fid=fopen('sre14_test.result','a');
fprintf(fid, '%6.3f,%6.3f,%6.3f,%6.3f,%6.3f,%6.3f\n', PLDA_eer,prog_PLDA_eer,eval_PLDA_eer,PLDA_dcf14,prog_PLDA_dcf14,eval_PLDA_dcf14);
fclose(fid);

% end
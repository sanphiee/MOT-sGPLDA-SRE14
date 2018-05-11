
%  ==========================================================================
%
%       author : Liang He, heliang@mail.tsinghua.edu.cn
%                Xianhong Chen, chenxianhong@mail.tsinghua.edu.cn
%   descrption : multiobjective optimization training 
%                simplified Gaussian probabilistic 
%                linear discriminant analysis (MOT, sGPLDA)
%                
%      created : 20180206
% last revised : 20180511
%
%    Liang He, +86-13426228839, heliang@mail.tsinghua.edu.cn
%    Aurora Lab, Department of Electronic Engineering, Tsinghua University
%  ==========================================================================

clc;
clear all;
close all;

load '../temp/sre14.mat';

% for wb_fac = 1.1:0.1:2

% for nphi = 50:50:200
    
nphi = 200; % dimension
niter = 50; % iteration
wb_fac = 1.7; % factor

%% train plda
plda = moplda_em_update_sigma(wb_fac, dev_ivec', dev_label, dev_ivec_neighbor', dev_label_neighbor, nphi, niter);

%% score plda
PLDA_scores = score_moplda_trials(plda, enrol_ivec', test_ivec');
PLDA_scores_col = reshape(PLDA_scores',length(PLDA_scores(:)),1);

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
% end
# MOT-sGPLDA-SRE14
Multiobjective Optimization Training of PLDA for Speaker Verification

1. prepare data, make directory ./data and ./temp, 
put NIST SRE14 offical data on "./data/", there are 
"development_data_labels.csv
dev_ivectors.csv
ivec14_sre_segment_key_release.tsv
ivec14_sre_trial_key_release.tsv
model_ivectors.csv
target_speaker_models.csv
test_ivectors.csv"

2. run ./python/sre14_preprocess.py.
It will generate "./temp/sre14.mat"

3. run ./matlab/gplda_demo.m.
The script will read "./temp/sre14.mat", and the results are
" 2.347, 2.456 (Dev, EER),  2.307 (Eval, EER), 
 0.264, 0.269 (Dev, MDCF), 0.261 (Eval, MDCF)". 

4. run ./matlab/moplda_demo.m.
The script will read "./temp/sre14.mat", and the results are
" 2.040, 2.193, 1.931, 0.233, 0.239, 0.229"

5. some experiment results.

A. train lambda with development and train vectors

0.fac exp, D EER, E EER,      , D DCF, E EER, factor
1.  2.531, 2.794, 2.354, 0.272, 0.277, 0.267, 1.1
2.  2.554, 2.825, 2.336, 0.269, 0.272, 0.266, 1.2
3.  2.456, 2.677, 2.176, 0.250, 0.255, 0.247, 1.3
4.  2.331, 2.579, 2.199, 0.238, 0.241, 0.235, 1.4
5.  2.207, 2.399, 2.099, 0.233, 0.235, 0.230, 1.5
6.  2.082, 2.272, 1.940, 0.230, 0.235, 0.225, 1.6
7.  2.040, 2.193, 1.931, 0.233, 0.239, 0.229, 1.7
8.  2.057, 2.180, 1.973, 0.238, 0.244, 0.232, 1.8
9.  2.136, 2.241, 2.049, 0.244, 0.250, 0.238, 1.9
10. 2.108, 2.261, 1.995, 0.242, 0.249, 0.237, 2.0

B. train lambda with development vectors

0.fac exp, D EER, E EER,      , D DCF, E EER, factor
1.  2.182, 2.426, 1.994, 0.241, 0.244, 0.237, 1.1
2.  2.257, 2.438, 2.099, 0.237, 0.243, 0.232, 1.2
3.  2.369, 2.487, 2.225, 0.240, 0.245, 0.236, 1.3
4.  2.307, 2.426, 2.210, 0.238, 0.245, 0.233, 1.4
5.  2.481, 2.610, 2.330, 0.264, 0.273, 0.257, 1.5
6.  2.340, 2.425, 2.254, 0.253, 0.260, 0.246, 1.6
7.  2.347, 2.487, 2.267, 0.253, 0.260, 0.247, 1.7
8.  2.337, 2.395, 2.301, 0.264, 0.272, 0.257, 1.8
9.  2.406, 2.456, 2.351, 0.266, 0.274, 0.260, 1.9
10. 2.506, 2.549, 2.435, 0.275, 0.281, 0.269, 2.0

C.  train lambda with development and train vectors

0.dim exp, D EER, E EER,      , D DCF, E EER, factor
1.  3.522, 3.556, 3.505, 0.498, 0.507, 0.491, 50
2.  2.306, 2.329, 2.267, 0.288, 0.294, 0.284, 100
3.  2.032, 2.241, 1.919, 0.239, 0.245, 0.234, 150
4.  2.040, 2.193, 1.931, 0.233, 0.239, 0.229, 200
5.  2.070, 2.260, 1.918, 0.234, 0.240, 0.229, 250

PS:
1. The sGPLDA demo was downloaded from https://github.com/wangwei2009/MSR-Identity-Toolkit-v1.0
2. Anaconda3, Python3, require sklearn
3. Matlab R2016a

function plda = moplda_em_update_sigma(wb_fac, data, spk_labs, data_diff, spk_labs_diff, nphi, niter)

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

[ndim, nobs] = size(data);
[ndim_diff, nobs_diff] = size(data_diff);

if ( nobs ~= length(spk_labs) ),
	error('oh dear! number of data samples should match the number of labels!');
end

if ( nobs_diff ~= length(spk_labs_diff) ),
	error('oh dear! number of data samples should match the number of labels!');
end

if ( ndim ~= ndim_diff ),
	error('oh dear! dim same!');
end

% make sure the labels are sorted
[spk_labs, I] = sort(spk_labs);
data = data(:, I);
[~, ia, ic] = unique(spk_labs, 'stable');
spk_counts = histc(ic, 1 : numel(ia)); % # sessions per speaker

% make sure the labels are sorted
[spk_labs_diff, I_diff] = sort(spk_labs_diff);
data_diff = data_diff(:, I_diff);
[~, ia_diff, ic_diff] = unique(spk_labs_diff, 'stable');
spk_counts_diff = histc(ic_diff, 1 : numel(ia_diff)); % # sessions per speaker

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % normalize
% M = mean(data, 2);
% data = bsxfun(@minus, data, M); % centering the data
% W1   = calc_white_mat(cov(data'));
% data = W1' * data; % whitening the data
% data = length_norm(data); % normalizing the length
% 
% data_diff = bsxfun(@minus, data_diff, M); % centering the data
% data_diff = W1' * data_diff; % whitening the data
% data_diff = length_norm(data_diff); % normalizing the length

fprintf('\n\nRandomly initializing the PLDA hyperparameters ...\n\n');
% Initialize the parameters randomly
[s1, s2] = RandStream.create('mrg32k3a', 'NumStreams', 2);
Sigma    = 100 * randn(s1, ndim); % covariance matrix of the residual term
Sigma_diff = 100 * randn(s1, ndim); % covariance matrix of the residual term
Phi = randn(s2, ndim, nphi); % factor loading matrix (Eignevoice matrix)
Phi = bsxfun(@minus, Phi, mean(Phi, 2));
W2  = calc_white_mat(Phi' * Phi);
Phi = Phi * W2; % orthogonalize Eigenvoices (columns)

fprintf('Re-estimating the Eigenvoice subspace with %d factors ...\n', nphi);
for iter = 1 : niter
    fprintf('EM iter#: %d \t', iter);
    tim = tic;
    
    % expectation
    [Ey, Eyy] = expectation_plda(data, Phi, Sigma, spk_counts);
    [Ey_diff, Eyy_diff] = expectation_plda(data_diff, Phi, Sigma_diff, spk_counts_diff);
    
    % maximization
    [Phi, Sigma, Sigma_diff] = maximization_plda(wb_fac, data, data_diff, Ey, Eyy, Ey_diff, Eyy_diff);
    tim = toc(tim);
    fprintf('[elaps = %.2f s]\n', tim);
end

plda.Phi   = Phi;
plda.Sigma = Sigma;
plda.Sigma_diff = Sigma_diff;
% plda.W     = W1;
% plda.M     = M;

function [Ey, Eyy] = expectation_plda(data, Phi, Sigma, spk_counts)
% computes the posterior mean and covariance of the factors
nphi     = size(Phi, 2);
nsamples = size(data, 2);
nspks    = size(spk_counts, 1);

Ey  = zeros(nphi, nsamples);
Eyy = zeros(nphi);

% initialize common terms to save computations
uniqFreqs  	  = unique(spk_counts);
nuniq 		  = size(uniqFreqs, 1);
invTerms      = cell(nuniq, 1);
invTerms(:)   = {zeros(nphi)};
PhiT_invS_Phi = ( Phi'/Sigma ) * Phi;
I = eye(nphi);
for ix = 1 : nuniq
    nPhiT_invS_Phi = uniqFreqs(ix) * PhiT_invS_Phi;
    Cyy =  pinv(I + nPhiT_invS_Phi);
    invTerms{ix} = Cyy;
end

data = Sigma\data;
cnt  = 1;
for spk = 1 : nspks
    nsessions = spk_counts(spk);
    % Speaker indices
    idx = cnt : ( cnt - 1 ) + spk_counts(spk);
    cnt  = cnt + spk_counts(spk);
    Data = data(:, idx);
    PhiT_invS_y = sum(Phi' * Data, 2);
    Cyy = invTerms{ uniqFreqs == nsessions };
    Ey_spk  = Cyy * PhiT_invS_y;
    Eyy_spk = Cyy + Ey_spk * Ey_spk';
    Eyy     = Eyy + nsessions * Eyy_spk;
    Ey(:, idx) = repmat(Ey_spk, 1, nsessions);
end


function [Phi, Sigma, Sigma_diff] = maximization_plda(wb_fac, data, data_diff, Ey, Eyy, Ey_diff, Eyy_diff)
% ML re-estimation of the Eignevoice subspace and the covariance of the
% residual noise (full).
nsamples = size(data, 2);
nsamples_diff = size(data_diff, 2);
Data_sqr = data * data';
Data_sqr_diff = data_diff * data_diff';
% Phi      = data * Ey' / (Eyy);
% Phi_diff = data_diff * Ey_diff' / (Eyy_diff);
A = (wb_fac/nsamples) * data * Ey';
B = (1/nsamples_diff) * data_diff * Ey_diff';
C = (wb_fac/nsamples) * Eyy;
D = (1/nsamples_diff) * Eyy_diff;
Phi      = (A - B) / (C - D);
Sigma    = 1/nsamples * (Data_sqr - (Phi * Ey) * data');
Sigma_diff = 1/nsamples_diff * (Data_sqr_diff - (Phi * Ey_diff) * data_diff');

% function [Phi, Sigma, Sigma_diff] = maximization_plda(wb_fac, data, data_diff, Ey, Eyy, Ey_diff, Eyy_diff)
% % ML re-estimation of the Eignevoice subspace and the covariance of the
% % residual noise (full).
% nsamples = size(data, 2);
% nsamples_diff = size(data_diff, 2);
% Data_sqr = data * data';
% Data_diff_sqr = data_diff * data_diff';
% % Phi      = data * Ey' / (Eyy);
% % Phi_diff = data_diff * Ey_diff' / (Eyy_diff);
% wb_fac = 1;
% A = (wb_fac/nsamples) * data * Ey';
% B = (1/nsamples_diff) * data_diff * Ey_diff';
% C = (wb_fac/nsamples) * Eyy;
% D = (1/nsamples_diff) * Eyy_diff;
% Phi_1 = A / C;
% Phi_2 = B / D;
% C1 = Phi_1 * Phi_1';
% D1 = Phi_2 * Phi_2';
% 
% [ev, ~] = eig(C1, D1);
% dim = size(Phi_1, 2);
% Phi      = ev(:,1:dim);
% Sigma    = 1/nsamples * (Data_sqr - (Phi * Ey) * data');
% Sigma_diff    = 1/nsamples * (Data_diff_sqr - (Phi * Ey_diff) * data_diff');


function W = calc_white_mat(X)
% calculates the whitening transformation for cov matrix X
[~, D, V] = svd(X);
W = V * diag(sparse(1./( sqrt(diag(D)) + 1e-10 )));

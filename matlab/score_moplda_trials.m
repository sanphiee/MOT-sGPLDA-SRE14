function scores = score_moplda_trials(plda, model_iv, test_iv)

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

if ~isstruct(plda),
	fprintf(1, 'Error: plda should be a structure!\n');
	return;
end

Phi     = plda.Phi;
Sigma   = plda.Sigma;
Sigma_diff   = plda.Sigma_diff;
% W       = plda.W;
% M       = plda.M;

% %%%%% post-processing the model i-vectors %%%%%
% model_iv = bsxfun(@minus, model_iv, M); % centering the data
% model_iv = W' * model_iv; % whitening data
% model_iv = length_norm(model_iv); % normalizing the length
% 
% %%%%% post-processing the test i-vectors %%%%%
% test_iv = bsxfun(@minus, test_iv, M); % centering the data
% test_iv  = W' * test_iv; % whitening data
% test_iv = length_norm(test_iv); % normalizing the length

nphi = size(Phi, 1); 

Sigma_ac  = Phi * Phi';
Sigma_tot = Sigma_ac + Sigma;
Sigma_tot_diff = Sigma_ac + Sigma_diff; % same, better

Sigma_tot_i = pinv(Sigma_tot);
Sigma_tot_diff_i = pinv(Sigma_tot_diff);

Sigma_i = pinv(Sigma_tot - Sigma_ac*Sigma_tot_i*Sigma_ac);
Q = Sigma_tot_diff_i - Sigma_i;
P = (Sigma_tot_i * Sigma_ac) * Sigma_i;

[U, S] = svd(P);
S = diag(S);
Lambda = diag(S(1 : nphi));
Uk     = U(:, 1 : nphi);
Q_hat  = Uk' * Q * Uk;

model_iv = Uk' * model_iv;
test_iv  = Uk' * test_iv;

score_h1 = diag(model_iv' * Q_hat * model_iv);
score_h2 = diag(test_iv' * Q_hat * test_iv);
score_h1h2 = 2 * model_iv' * Lambda * test_iv;

scores = bsxfun(@plus, score_h1h2, score_h1);
scores = bsxfun(@plus, scores, score_h2');




% function scores = score_dplda_trials(plda, model_iv, test_iv)
% % computes the verification scores as the log-likelihood ratio of the same 
% % versus different speaker models hypotheses.
% %
% % Inputs:
% %   plda            : structure containing PLDA hyperparameters
% %   model_iv        : data matrix for enrollment i-vectors (column observations)
% %   test_iv         : data matrix for test i-vectors (one observation per column)
% %
% % Outputs:
% %    scores         : output verification scores matrix (all model-test combinations)
% %
% % References:
% %   [1] D. Garcia-Romero and C.Y. Espy-Wilson, "Analysis of i-vector length 
% %       normalization in speaker recognition systems," in Proc. INTERSPEECH,
% %       Florence, Italy, Aug. 2011, pp. 249-252.
% %
% % Omid Sadjadi <s.omid.sadjadi@gmail.com>
% % Microsoft Research, Conversational Systems Research Center
% 
% if ~isstruct(plda),
% 	fprintf(1, 'Error: plda should be a structure!\n');
% 	return;
% end
% 
% Phi     = plda.Phi;
% Sigma   = plda.Sigma;
% Sigma_diff   = plda.Sigma_diff;
% W       = plda.W;
% M       = plda.M;
% 
% %%%%% post-processing the model i-vectors %%%%%
% model_iv = bsxfun(@minus, model_iv, M); % centering the data
% model_iv = W' * model_iv; % whitening data
% model_iv = length_norm(model_iv); % normalizing the length
% 
% %%%%% post-processing the test i-vectors %%%%%
% test_iv = bsxfun(@minus, test_iv, M); % centering the data
% test_iv  = W' * test_iv; % whitening data
% test_iv = length_norm(test_iv); % normalizing the length
% 
% scores1 = eval_score(Phi, Sigma, model_iv, test_iv);
% scores2 = eval_score(Phi, Sigma_diff, model_iv, test_iv);
% scores = scores1 - scores2;
% 
% function scores = eval_score(Phi, Sigma, model_iv, test_iv)
% nphi = size(Phi, 1); 
% 
% Sigma_ac  = Phi * Phi';
% Sigma_tot = Sigma_ac + Sigma;
% 
% Sigma_tot_i = pinv(Sigma_tot);
% Sigma_i = pinv(Sigma_tot - Sigma_ac*Sigma_tot_i*Sigma_ac);
% 
% Q = - Sigma_i;
% P = (Sigma_tot_i * Sigma_ac) * Sigma_i;
% 
% [U, S] = svd(P);
% S = diag(S);
% Lambda = diag(S(1 : nphi));
% Uk     = U(:, 1 : nphi);
% Q_hat  = Uk' * Q * Uk;
% 
% model_iv = Uk' * model_iv;
% test_iv  = Uk' * test_iv;
% 
% score_h1 = diag(model_iv' * Q_hat * model_iv);
% score_h2 = diag(test_iv' * Q_hat * test_iv);
% score_h1h2 = 2 * model_iv' * Lambda * test_iv;
% 
% scores = bsxfun(@plus, score_h1h2, score_h1);
% scores = bsxfun(@plus, scores, score_h2');
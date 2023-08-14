%% Collect and plot SBD results

addpath(genpath('../seal-master/lib/matlab/'));

categories = categories_city();

% Original GT (Thin)
result_dir = {
    './scores_orig_thin_sbd_inst_casenet-s'; 
};

plot_pr(result_dir, {'STEAL'},...
         './output/scores/pr_curves', categories, false);
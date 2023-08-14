addpath(genpath('../seal-master/lib/matlab/'));

categories = categories_sbd();

eval_dir = {
    './sbd_inst_casenet-s'; 
};

result_dir = {
    './scores_reanno_thin_sbd_inst_casenet-s'; 
}

evaluation('./sbd-preprocess/gt_eval/gt_orig_thin/test.mat',...
           './sbd-preprocess/gt_eval/gt_reanno_thin/inst',...
           eval_dir, result_dir, categories, 5, 99, true, 0.0075)
           
           

eval_dir = {
    './sbd_inst_casenet-s'; 
};

result_dir = {
    './scores_orig_thin_sbd_inst_casenet-s'; 
};

evaluation('./sbd-preprocess/gt_eval/gt_orig_thin/test.mat',...
           '/sbd-preprocess/gt_eval/gt_orig_thin/inst',...
           eval_dir, result_dir, categories, 5, 99, true, 0.02)
%% Create configuration object of class 'coder.CodeConfig'.
cfg = coder.config('lib','ecoder',false);
cfg.GenerateReport = true;
cfg.ReportPotentialDifferences = true;
cfg.EnableRuntimeRecursion = false;
cfg.DynamicMemoryAllocation = 'Off';
cfg.SaturateOnIntegerOverflow = false;
cfg.FilePartitionMethod = 'SingleFile';
cfg.EnableMemcpy = false;
cfg.PreserveVariableNames = 'All';
cfg.RowMajor = true;
cfg.EnableOpenMP = false;
cfg.GenCodeOnly = true;
cfg.SupportNonFinite = false;

%% Define argument types for entry-point 'eTpred_prep_inputs'.
ARGS = cell(1,1);
ARGS{1} = cell(4,1);
ARGS{1}{1} = coder.typeof(single(0));
ARGS{1}{2} = coder.typeof(single(0));
ARGS{1}{3} = coder.typeof(single(0),[1 100],[0 1]);
ARGS{1}{4} = coder.typeof(single(0),[1 100],[0 1]);

%% Invoke MATLAB Coder.
codegen -config cfg -singleC eTpred_prep_inputs -args ARGS{1}


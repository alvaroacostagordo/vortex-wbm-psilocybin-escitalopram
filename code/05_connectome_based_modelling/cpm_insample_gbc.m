function CPM = cpm_insample_gbc(data_source, cohort, psilocybin_dir, escitalopram_dir, bdi_psilocybin, bdi_escitalopram, output_file)
% Descriptive in-sample association between baseline GBC and BDI improvement.

% The analysis is performed separately in signal or vortex space. Individual
% clinical outcomes and connectivity matrices are supplied at runtime and are
% not embedded in the analysis code.

% Example:
% CPM = cpm_insample_gbc('signal', 'all', signal_psilo_dir, ...
%     signal_escitalopram_dir, bdi_psilocybin, bdi_escitalopram);

% BDI improvement is defined as pre-treatment minus post-treatment score, so
% larger positive values indicate greater clinical improvement.

% This is an explicitly in-sample, descriptive analysis. The fitted
% correlations and nominal model P values must not be interpreted as
% out-of-sample predictive performance.

if nargin < 6
    error(['Required inputs: data_source, cohort, psilocybin_dir, ' ...
        'escitalopram_dir, bdi_psilocybin, and bdi_escitalopram.']);
end

data_source = lower(strtrim(char(data_source)));
cohort = lower(strtrim(char(cohort)));

switch data_source
    case 'signal'
        file_pattern = 'model_hopf_FC_*.mat';
    case 'vortex'
        file_pattern = 'model_vortex_partition_social_*.mat';
    otherwise
        error('Unknown data_source. Use ''signal'' or ''vortex''.');
end

variable_name = 'Ceff_task1';
thresholds = 0.10:0.02:0.60;

bdi_psilocybin = bdi_psilocybin(:);
bdi_escitalopram = bdi_escitalopram(:);

GEC_pre_psilocybin = load_connectivity_matrices( ...
    psilocybin_dir, file_pattern, variable_name, numel(bdi_psilocybin));
GEC_pre_escitalopram = load_connectivity_matrices( ...
    escitalopram_dir, file_pattern, variable_name, numel(bdi_escitalopram));

switch cohort
    case 'psilocybin'
        GEC_pre = GEC_pre_psilocybin;
        y = bdi_psilocybin;
    case 'escitalopram'
        GEC_pre = GEC_pre_escitalopram;
        y = bdi_escitalopram;
    case 'all'
        GEC_pre = cat(1, GEC_pre_psilocybin, GEC_pre_escitalopram);
        y = [bdi_psilocybin; bdi_escitalopram];
    otherwise
        error(['Unknown cohort. Use ''psilocybin'', ''escitalopram'', ' ...
            'or ''all''.']);
end

if any(~isfinite(y))
    error('BDI improvement values must be finite.');
end

X = compute_gbc(GEC_pre);

if any(sum(abs(X), 2) == 0)
    error('At least one participant has all-zero GBC values.');
end

n_thresholds = numel(thresholds);
n_subjects = size(X, 1);
n_nodes = size(X, 2);

feature_correlations = nan(1, n_nodes);
for node = 1:n_nodes
    feature_correlations(node) = corr( ...
        X(:, node), y, 'Rows', 'complete');
end
feature_correlations(~isfinite(feature_correlations)) = 0;

fitted_correlation = nan(n_thresholds, 1);
R2 = nan(n_thresholds, 1);
P_model_nominal = nan(n_thresholds, 1);
n_positive = zeros(n_thresholds, 1);
n_negative = zeros(n_thresholds, 1);
fitted_values = nan(n_subjects, n_thresholds);

positive_masks = cell(n_thresholds, 1);
negative_masks = cell(n_thresholds, 1);
model_coefficients = cell(n_thresholds, 1);
design_columns = cell(n_thresholds, 1);

for threshold_index = 1:n_thresholds
    threshold = thresholds(threshold_index);

    positive_mask = feature_correlations > threshold;
    negative_mask = feature_correlations < -threshold;

    positive_masks{threshold_index} = positive_mask;
    negative_masks{threshold_index} = negative_mask;
    n_positive(threshold_index) = sum(positive_mask);
    n_negative(threshold_index) = sum(negative_mask);

    if n_positive(threshold_index) + n_negative(threshold_index) == 0
        fprintf('Threshold %.2f | no GBC features selected\n', threshold);
        continue
    end

    positive_score = sum(X(:, positive_mask), 2);
    negative_score = sum(X(:, negative_mask), 2);
    design = [positive_score, negative_score];

    keep_columns = std(design, 0, 1) > 0;
    design = design(:, keep_columns);

    if isempty(design)
        fprintf(['Threshold %.2f | selected GBC features produced ' ...
            'zero-variance scores\n'], threshold);
        continue
    end

    model = fitlm(design, y);
    y_fitted = predict(model, design);

    fitted_values(:, threshold_index) = y_fitted;
    fitted_correlation(threshold_index) = corr( ...
        y, y_fitted, 'Rows', 'complete');
    R2(threshold_index) = model.Rsquared.Ordinary;
    P_model_nominal(threshold_index) = coefTest(model);
    model_coefficients{threshold_index} = model.Coefficients.Estimate;
    design_columns{threshold_index} = keep_columns;

    fprintf(['Threshold %.2f | positive %d | negative %d | ' ...
        'fitted r %.3f | R2 %.3f | nominal model P %.4f\n'], ...
        threshold, n_positive(threshold_index), ...
        n_negative(threshold_index), ...
        fitted_correlation(threshold_index), R2(threshold_index), ...
        P_model_nominal(threshold_index));
end

valid_thresholds = find(isfinite(fitted_correlation));
if isempty(valid_thresholds)
    error('No threshold produced a valid in-sample model.');
end

[best_correlation, valid_index] = max( ...
    fitted_correlation(valid_thresholds));
best_index = valid_thresholds(valid_index);

CPM = struct();
CPM.analysis = 'descriptive in-sample GBC association';
CPM.data_source = data_source;
CPM.cohort = cohort;
CPM.thresholds = thresholds;
CPM.feature_correlations = feature_correlations;
CPM.fitted_correlation = fitted_correlation;
CPM.R2 = R2;
CPM.P_model_nominal = P_model_nominal;
CPM.n_positive = n_positive;
CPM.n_negative = n_negative;
CPM.positive_masks = positive_masks;
CPM.negative_masks = negative_masks;
CPM.model_coefficients = model_coefficients;
CPM.design_columns = design_columns;
CPM.fitted_values = fitted_values;
CPM.best_index = best_index;
CPM.best_threshold = thresholds(best_index);
CPM.best_correlation = best_correlation;
CPM.best_R2 = R2(best_index);
CPM.best_P_model_nominal = P_model_nominal(best_index);
CPM.best_fitted_values = fitted_values(:, best_index);

fprintf('\nBest descriptive in-sample model:\n');
fprintf(['Threshold %.2f | fitted r %.3f | R2 %.3f | ' ...
    'nominal model P %.4f | positive %d | negative %d\n'], ...
    CPM.best_threshold, CPM.best_correlation, CPM.best_R2, ...
    CPM.best_P_model_nominal, n_positive(best_index), ...
    n_negative(best_index));

if nargin < 7 || isempty(output_file)
    output_file = sprintf( ...
        'cpm_insample_gbc_%s_%s.mat', data_source, cohort);
end

save(output_file, 'CPM', '-v7.3');
fprintf('Saved results to %s\n', output_file);

end


function matrices = load_connectivity_matrices(directory, file_pattern, ...
    variable_name, expected_subjects)

directory = char(directory);
if exist(directory, 'dir') ~= 7
    error('Connectivity directory does not exist: %s', directory);
end

files = dir(fullfile(directory, file_pattern));
if isempty(files)
    error('No files found in %s with pattern %s.', directory, file_pattern);
end

[~, order] = sort({files.name});
files = files(order);

if numel(files) ~= expected_subjects
    error(['Expected %d connectivity files in %s to match the clinical ' ...
        'outcomes, but found %d.'], ...
        expected_subjects, directory, numel(files));
end

matrices = [];
for subject = 1:numel(files)
    file_path = fullfile(files(subject).folder, files(subject).name);
    loaded = load(file_path, variable_name);

    if ~isfield(loaded, variable_name)
        error('Variable %s was not found in %s.', variable_name, file_path);
    end

    matrix = double(loaded.(variable_name));
    if ndims(matrix) ~= 2 || size(matrix, 1) ~= size(matrix, 2)
        error('Connectivity matrix in %s must be square.', file_path);
    end
    if any(~isfinite(matrix(:)))
        error('Non-finite connectivity values found in %s.', file_path);
    end

    if subject == 1
        matrices = zeros( ...
            numel(files), size(matrix, 1), size(matrix, 2));
    elseif size(matrix, 1) ~= size(matrices, 2)
        error('Connectivity matrices do not have consistent dimensions.');
    end

    matrices(subject, :, :) = matrix;
end


end


function GBC = compute_gbc(matrices)

n_subjects = size(matrices, 1);
n_nodes = size(matrices, 2);

if size(matrices, 3) ~= n_nodes || n_nodes < 2
    error('Connectivity matrices must be square and contain at least two nodes.');
end

GBC = zeros(n_subjects, n_nodes);
diagonal_indices = 1:(n_nodes + 1):(n_nodes * n_nodes);

for subject = 1:n_subjects
    matrix = squeeze(matrices(subject, :, :));
    matrix(diagonal_indices) = 0;
    GBC(subject, :) = sum(matrix, 1) / (n_nodes - 1);
end


end

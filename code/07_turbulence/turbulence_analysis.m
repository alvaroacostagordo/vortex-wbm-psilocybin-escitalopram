function results = turbulence_analysis( ...
    psilocybin_pre_dir, psilocybin_post_dir, ...
    escitalopram_pre_dir, escitalopram_post_dir, ...
    delta_bdi_psilocybin, delta_bdi_escitalopram, ...
    responders_psilocybin, responders_escitalopram, output_dir)
% Analyse longitudinal turbulence changes and clinical associations.

% Turbulence is averaged within each participant and session. Longitudinal
% change among responders pooled across treatments is evaluated using a
% two-sided Wilcoxon signed-rank test with 10,000 sign-flip permutations.
% Clinical associations are evaluated in each treatment group and in the
% combined sample using Pearson correlations between post-minus-pre changes
% in turbulence and BDI.

% delta_bdi_psilocybin and delta_bdi_escitalopram must contain post-minus-pre
% BDI changes, such that negative values indicate clinical improvement.
% The responder inputs must be logical vectors aligned with the participants.

% Example:
% results = turbulence_analysis( ...
%     psilocybin_pre_dir, psilocybin_post_dir, ...
%     escitalopram_pre_dir, escitalopram_post_dir, ...
%     delta_bdi_psilocybin, delta_bdi_escitalopram, ...
%     responders_psilocybin, responders_escitalopram);


if nargin < 8
    error(['Required inputs: four connectivity directories and the ' ...
        'BDI-change and responder-status vectors for both treatments.']);
end

if nargin < 9 || isempty(output_dir)
    output_dir = pwd;
end

if exist(output_dir, 'dir') ~= 7
    mkdir(output_dir);
end

file_pattern = 'model_vortex_partition_social_*.mat';
variable_name = 'Turbu_emp_task1';
n_permutations = 10000;
random_seed = 12345;

[turbulence_psilocybin_pre, names_psilocybin_pre] = ...
    load_turbulence_group( ...
        psilocybin_pre_dir, file_pattern, variable_name, 22);
[turbulence_psilocybin_post, names_psilocybin_post] = ...
    load_turbulence_group( ...
        psilocybin_post_dir, file_pattern, variable_name, 22);
[turbulence_escitalopram_pre, names_escitalopram_pre] = ...
    load_turbulence_group( ...
        escitalopram_pre_dir, file_pattern, variable_name, 20);
[turbulence_escitalopram_post, names_escitalopram_post] = ...
    load_turbulence_group( ...
        escitalopram_post_dir, file_pattern, variable_name, 20);

if ~isequal(names_psilocybin_pre, names_psilocybin_post)
    error('Psilocybin pre- and post-treatment filenames do not match.');
end

if ~isequal(names_escitalopram_pre, names_escitalopram_post)
    error('Escitalopram pre- and post-treatment filenames do not match.');
end

delta_bdi_psilocybin = delta_bdi_psilocybin(:);
delta_bdi_escitalopram = delta_bdi_escitalopram(:);

if numel(delta_bdi_psilocybin) ~= 22
    error('Expected 22 psilocybin BDI change scores; found %d.', ...
        numel(delta_bdi_psilocybin));
end


if numel(delta_bdi_escitalopram) ~= 20
    error('Expected 20 escitalopram BDI change scores; found %d.', ...
        numel(delta_bdi_escitalopram));
end

if any(~isfinite(delta_bdi_psilocybin)) || ...
        any(~isfinite(delta_bdi_escitalopram))
    error('BDI change scores must be finite.');
end

responders_psilocybin = validate_responder_mask( ...
    responders_psilocybin, 22, 18, 'psilocybin');
responders_escitalopram = validate_responder_mask( ...
    responders_escitalopram, 20, 8, 'escitalopram');

delta_turbulence_psilocybin = ...
    turbulence_psilocybin_post - turbulence_psilocybin_pre;
delta_turbulence_escitalopram = ...
    turbulence_escitalopram_post - turbulence_escitalopram_pre;

delta_turbulence_all = [ ...
    delta_turbulence_psilocybin; delta_turbulence_escitalopram];
delta_bdi_all = [delta_bdi_psilocybin; delta_bdi_escitalopram];

turbulence_responders_pre = [ ...
    turbulence_psilocybin_pre(responders_psilocybin); ...
    turbulence_escitalopram_pre(responders_escitalopram)];
turbulence_responders_post = [ ...
    turbulence_psilocybin_post(responders_psilocybin); ...
    turbulence_escitalopram_post(responders_escitalopram)];
delta_turbulence_responders = ...
    turbulence_responders_post - turbulence_responders_pre;

rng(random_seed, 'twister');
sign_mask_responders = rand( ...
    n_permutations, numel(turbulence_responders_pre)) > 0.5;

[p_responders, W_responders] = ...
    wilcoxon_signed_rank_permutation( ...
        turbulence_responders_pre, turbulence_responders_post, ...
        sign_mask_responders);

d_responders = cohens_dz(delta_turbulence_responders);

longitudinal_table = build_longitudinal_table( ...
    turbulence_responders_pre, turbulence_responders_post, ...
    W_responders, d_responders, p_responders);

[r_psilocybin, p_correlation_psilocybin] = corr( ...
    delta_turbulence_psilocybin, delta_bdi_psilocybin, ...
    'Type', 'Pearson');
[r_escitalopram, p_correlation_escitalopram] = corr( ...
    delta_turbulence_escitalopram, delta_bdi_escitalopram, ...
    'Type', 'Pearson');
[r_all, p_correlation_all] = corr( ...
    delta_turbulence_all, delta_bdi_all, 'Type', 'Pearson');

correlation_table = table( ...
    ["Psilocybin"; "Escitalopram"; "Combined"], ...
    [22; 20; 42], ...
    [r_psilocybin; r_escitalopram; r_all], ...
    [p_correlation_psilocybin; ...
        p_correlation_escitalopram; p_correlation_all], ...
    'VariableNames', {'Group', 'N', 'PearsonR', 'PValue'});

results = struct();
results.settings.turbulence_summary = ...
    'Mean of Turbu_emp_task1 within participant and session';
results.settings.turbulence_delta = 'post minus pre';
results.settings.bdi_delta = ...
    'post minus pre; negative values indicate improvement';
results.settings.longitudinal_test = ...
    ['Combined responders: two-sided Wilcoxon signed-rank with ' ...
    'sign-flip permutations'];
results.settings.n_permutations = n_permutations;
results.settings.random_seed = random_seed;
results.settings.correlation = 'Pearson';
results.longitudinal = longitudinal_table;
results.clinical_correlations = correlation_table;

mat_file = fullfile(output_dir, 'turbulence_analysis_results.mat');
longitudinal_file = fullfile( ...
    output_dir, 'turbulence_responders_longitudinal_statistics.csv');
correlation_file = fullfile( ...
    output_dir, 'turbulence_clinical_correlations.csv');

save(mat_file, 'results');
writetable(longitudinal_table, longitudinal_file);
writetable(correlation_table, correlation_file);

fprintf('\nLongitudinal turbulence changes\n');
disp(longitudinal_table);
fprintf('\nClinical correlations\n');
disp(correlation_table);
fprintf('Saved results to %s\n', output_dir);


end


function mask = validate_responder_mask( ...
    mask, expected_subjects, expected_responders, group_name)

mask = mask(:);
if numel(mask) ~= expected_subjects
    error('Expected %d %s responder-status values; found %d.', ...
        expected_subjects, group_name, numel(mask));
end

if ~islogical(mask)
    if ~isnumeric(mask) || any(~isfinite(mask)) || ...
            any(mask ~= 0 & mask ~= 1)
        error('%s responder status must contain only true/false or 1/0.', ...
            group_name);
    end
    mask = logical(mask);
end

if sum(mask) ~= expected_responders
    error('Expected %d %s responders; found %d.', ...
        expected_responders, group_name, sum(mask));
end


end


function [turbulence, sorted_names] = load_turbulence_group( ...
    directory, file_pattern, variable_name, expected_subjects)

directory = char(directory);
if exist(directory, 'dir') ~= 7
    error('Turbulence directory does not exist: %s', directory);
end

files = dir(fullfile(directory, file_pattern));
if numel(files) ~= expected_subjects
    error('Expected %d files in %s; found %d.', ...
        expected_subjects, directory, numel(files));
end

[~, order] = sort({files.name});
files = files(order);
sorted_names = string({files.name})';
turbulence = zeros(expected_subjects, 1);

for subject = 1:expected_subjects
    file_path = fullfile(files(subject).folder, files(subject).name);
    loaded = load(file_path, variable_name);

    if ~isfield(loaded, variable_name)
        error('Variable %s was not found in %s.', ...
            variable_name, file_path);
    end

    subject_turbulence = double(loaded.(variable_name));
    if isempty(subject_turbulence) || ...
            any(~isfinite(subject_turbulence(:)))
        error('Invalid turbulence values found in %s.', file_path);
    end

    turbulence(subject) = mean(subject_turbulence(:));
end


end


function [p_value, W_plus_observed] = ...
    wilcoxon_signed_rank_permutation(pre, post, sign_mask)

differences = post(:) - pre(:);
if any(~isfinite(differences))
    error('Non-finite values encountered in a paired comparison.');
end

nonzero = differences ~= 0;
differences = differences(nonzero);

if isempty(differences)
    W_plus_observed = 0;
    p_value = 1;
    return
end

ranks = tiedrank(abs(differences));
ranks = ranks(:);
W_plus_observed = sum(ranks(differences > 0));

expected_W = sum(ranks) / 2;
observed_statistic = abs(W_plus_observed - expected_W);

permuted_W = double(sign_mask(:, nonzero)) * ranks;
permuted_statistics = abs(permuted_W - expected_W);

tolerance = 1e-12;
p_value = (sum(permuted_statistics >= ...
    observed_statistic - tolerance) + 1) / (size(sign_mask, 1) + 1);


end


function d = cohens_dz(differences)

differences = differences(:);
mean_difference = mean(differences);
standard_deviation = std(differences, 0);

if standard_deviation == 0
    if mean_difference == 0
        d = 0;
    else
        d = sign(mean_difference) * Inf;
    end
else
    d = mean_difference / standard_deviation;
end


end


function longitudinal_table = build_longitudinal_table( ...
    pre_values, post_values, statistics, effect_sizes, p_values)

group = "Combined responders";
delta = post_values - pre_values;
n_subjects = numel(delta);
mean_pre = mean(pre_values);
standard_deviation_pre = std(pre_values);
mean_post = mean(post_values);
standard_deviation_post = std(post_values);
mean_delta = mean(delta);
median_delta = median(delta);

longitudinal_table = table( ...
    group, n_subjects, mean_pre, standard_deviation_pre, ...
    mean_post, standard_deviation_post, mean_delta, median_delta, ...
    statistics, effect_sizes, p_values, ...
    'VariableNames', { ...
    'Group', 'N', 'MeanPre', 'SDPre', 'MeanPost', 'SDPost', ...
    'MeanDelta', 'MedianDelta', 'WilcoxonWPlus', ...
    'CohensDz', 'PermutationP'});


end

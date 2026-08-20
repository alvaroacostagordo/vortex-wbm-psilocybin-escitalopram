function results = newman_modularity_analysis(data_source, ...
    psilocybin_pre_dir, psilocybin_post_dir, ...
    escitalopram_pre_dir, escitalopram_post_dir, output_dir)
% Compute Newman modularity and test treatment-related changes.

% The function runs the Louvain algorithm 100 times for each connectivity
% matrix and retains the partition with the highest Newman modularity Q.
% It analyses Q and the number of modules using Wilcoxon tests with 10,000
% Monte Carlo permutations.

% Example:
% results = newman_modularity_analysis('vortex', ...
%     psilocybin_pre_dir, psilocybin_post_dir, ...
%     escitalopram_pre_dir, escitalopram_post_dir, 'results');

% Requires community_louvain.m from the Brain Connectivity Toolbox.


if nargin < 5
    error(['Required inputs: data_source and the four connectivity ' ...
        'directories.']);
end

if nargin < 6 || isempty(output_dir)
    output_dir = pwd;
end

if exist('community_louvain', 'file') ~= 2
    error(['community_louvain.m was not found. Add the Brain ' ...
        'Connectivity Toolbox to the MATLAB path.']);
end

data_source = lower(strtrim(char(data_source)));
output_dir = char(output_dir);

switch data_source
    case 'signal'
        file_pattern = 'model_hopf_FC_*.mat';
        mat_filename = 'modularity_signal_results.mat';
    case 'vortex'
        file_pattern = 'model_vortex_partition_social_*.mat';
        mat_filename = 'modularity_results.mat';
    otherwise
        error('Unknown data_source. Use ''signal'' or ''vortex''.');
end

if exist(output_dir, 'dir') ~= 7
    mkdir(output_dir);
end

variable_name = 'Ceff_task1';
gamma = 1;
n_louvain_iterations = 100;
n_permutations = 10000;
louvain_seed = 42;
permutation_seed = 12345;

rng(louvain_seed, 'twister');

[Q_group_psilo_before, M_group_psilo_before, names_psilo_before] = ...
    compute_modularity_group(psilocybin_pre_dir, file_pattern, ...
        variable_name, 22, gamma, n_louvain_iterations);
[Q_group_psilo_after, M_group_psilo_after, names_psilo_after] = ...
    compute_modularity_group(psilocybin_post_dir, file_pattern, ...
        variable_name, 22, gamma, n_louvain_iterations);
[Q_group_esc_before, M_group_esc_before, names_esc_before] = ...
    compute_modularity_group(escitalopram_pre_dir, file_pattern, ...
        variable_name, 20, gamma, n_louvain_iterations);
[Q_group_esc_after, M_group_esc_after, names_esc_after] = ...
    compute_modularity_group(escitalopram_post_dir, file_pattern, ...
        variable_name, 20, gamma, n_louvain_iterations);

if ~isequal(names_psilo_before, names_psilo_after)
    error('Psilocybin pre- and post-treatment filenames do not match.');
end

if ~isequal(names_esc_before, names_esc_after)
    error('Escitalopram pre- and post-treatment filenames do not match.');
end

num_modules_psilo_before = count_modules(M_group_psilo_before);
num_modules_psilo_after = count_modules(M_group_psilo_after);
num_modules_esc_before = count_modules(M_group_esc_before);
num_modules_esc_after = count_modules(M_group_esc_after);

delta_Q_psilo = Q_group_psilo_after - Q_group_psilo_before;
delta_Q_esc = Q_group_esc_after - Q_group_esc_before;
delta_modules_psilo = ...
    num_modules_psilo_after - num_modules_psilo_before;
delta_modules_esc = ...
    num_modules_esc_after - num_modules_esc_before;

rng(permutation_seed, 'twister');

sign_mask_psilo = rand(n_permutations, 22) > 0.5;
sign_mask_esc = rand(n_permutations, 20) > 0.5;

group_mask = false(n_permutations, 42);
for permutation = 1:n_permutations
    selected = randperm(42, 20);
    group_mask(permutation, selected) = true;
end

[p_Q_esc, W_Q_esc] = wilcoxon_signed_rank_permutation( ...
    Q_group_esc_before, Q_group_esc_after, sign_mask_esc);
[p_Q_psilo, W_Q_psilo] = wilcoxon_signed_rank_permutation( ...
    Q_group_psilo_before, Q_group_psilo_after, sign_mask_psilo);
[p_Q_between, W_Q_between] = wilcoxon_rank_sum_permutation( ...
    delta_Q_esc, delta_Q_psilo, group_mask);

[p_modules_esc, W_modules_esc] = ...
    wilcoxon_signed_rank_permutation( ...
        num_modules_esc_before, num_modules_esc_after, sign_mask_esc);
[p_modules_psilo, W_modules_psilo] = ...
    wilcoxon_signed_rank_permutation( ...
        num_modules_psilo_before, num_modules_psilo_after, ...
        sign_mask_psilo);
[p_modules_between, W_modules_between] = ...
    wilcoxon_rank_sum_permutation( ...
        delta_modules_esc, delta_modules_psilo, group_mask);

d_Q_esc = cohens_dz(delta_Q_esc);
d_Q_psilo = cohens_dz(delta_Q_psilo);
d_Q_between = cohens_d_independent(delta_Q_esc, delta_Q_psilo);

d_modules_esc = cohens_dz(delta_modules_esc);
d_modules_psilo = cohens_dz(delta_modules_psilo);
d_modules_between = cohens_d_independent( ...
    delta_modules_esc, delta_modules_psilo);

statistics_table = build_statistics_table( ...
    Q_group_esc_before, Q_group_esc_after, ...
    Q_group_psilo_before, Q_group_psilo_after, ...
    num_modules_esc_before, num_modules_esc_after, ...
    num_modules_psilo_before, num_modules_psilo_after, ...
    [W_Q_esc; W_Q_psilo; W_Q_between; ...
        W_modules_esc; W_modules_psilo; W_modules_between], ...
    [d_Q_esc; d_Q_psilo; d_Q_between; ...
        d_modules_esc; d_modules_psilo; d_modules_between], ...
    [p_Q_esc; p_Q_psilo; p_Q_between; ...
        p_modules_esc; p_modules_psilo; p_modules_between]);

results = struct();
results.settings.data_source = data_source;
results.settings.modularity = 'Newman';
results.settings.gamma = gamma;
results.settings.n_louvain_iterations = n_louvain_iterations;
results.settings.n_permutations = n_permutations;
results.settings.louvain_seed = louvain_seed;
results.settings.permutation_seed = permutation_seed;
results.settings.delta_definition = 'post minus pre';
results.settings.between_contrast = ...
    'escitalopram delta minus psilocybin delta';

results.psilocybin.filenames_pre = names_psilo_before;
results.psilocybin.filenames_post = names_psilo_after;
results.psilocybin.Q_pre = Q_group_psilo_before;
results.psilocybin.Q_post = Q_group_psilo_after;
results.psilocybin.delta_Q = delta_Q_psilo;
results.psilocybin.M_pre = M_group_psilo_before;
results.psilocybin.M_post = M_group_psilo_after;
results.psilocybin.n_modules_pre = num_modules_psilo_before;
results.psilocybin.n_modules_post = num_modules_psilo_after;
results.psilocybin.delta_n_modules = delta_modules_psilo;

results.escitalopram.filenames_pre = names_esc_before;
results.escitalopram.filenames_post = names_esc_after;
results.escitalopram.Q_pre = Q_group_esc_before;
results.escitalopram.Q_post = Q_group_esc_after;
results.escitalopram.delta_Q = delta_Q_esc;
results.escitalopram.M_pre = M_group_esc_before;
results.escitalopram.M_post = M_group_esc_after;
results.escitalopram.n_modules_pre = num_modules_esc_before;
results.escitalopram.n_modules_post = num_modules_esc_after;
results.escitalopram.delta_n_modules = delta_modules_esc;

results.statistics = statistics_table;

mat_file = fullfile(output_dir, mat_filename);
csv_file = fullfile(output_dir, sprintf( ...
    'modularity_statistics_%s.csv', data_source));

save(mat_file, ...
    'Q_group_esc_before', 'M_group_esc_before', ...
    'Q_group_esc_after', 'M_group_esc_after', ...
    'Q_group_psilo_before', 'M_group_psilo_before', ...
    'Q_group_psilo_after', 'M_group_psilo_after', ...
    'num_modules_esc_before', 'num_modules_esc_after', ...
    'num_modules_psilo_before', 'num_modules_psilo_after', ...
    'statistics_table', 'results');
writetable(statistics_table, csv_file);

fprintf('\nNewman modularity analysis | %s space\n', upper(data_source));
disp(statistics_table(:, { ...
    'Measure', 'Comparison', 'MeanAMinusB', ...
    'EffectSize', 'PermutationP'}));
fprintf('Saved results to %s\n', output_dir);


end


function [Q_group, M_group, sorted_names] = ...
    compute_modularity_group(data_dir, file_pattern, variable_name, ...
        expected_subjects, gamma, n_iterations)

data_dir = char(data_dir);
if exist(data_dir, 'dir') ~= 7
    error('Connectivity directory does not exist: %s', data_dir);
end

files = dir(fullfile(data_dir, file_pattern));
if numel(files) ~= expected_subjects
    error('Expected %d files in %s; found %d.', ...
        expected_subjects, data_dir, numel(files));
end

[~, order] = sort({files.name});
files = files(order);
sorted_names = string({files.name})';

Q_group = nan(1, expected_subjects);
M_group = [];

for subject = 1:expected_subjects
    file_path = fullfile(files(subject).folder, files(subject).name);
    loaded = load(file_path, variable_name);

    if ~isfield(loaded, variable_name)
        error('Variable %s was not found in %s.', ...
            variable_name, file_path);
    end

    W = double(loaded.(variable_name))';
    if ndims(W) ~= 2 || size(W, 1) ~= size(W, 2)
        error('Connectivity matrix in %s must be square.', file_path);
    end
    if any(~isfinite(W(:)))
        error('Non-finite connectivity values found in %s.', file_path);
    end

    W(1:(size(W, 1) + 1):end) = 0;

    if any(W(:) < -1e-10)
        error(['Newman modularity requires non-negative weights. ' ...
            'Negative values were found in %s.'], file_path);
    end

    if subject == 1
        M_group = zeros(expected_subjects, size(W, 1));
    elseif size(W, 1) ~= size(M_group, 2)
        error('Connectivity matrices do not have consistent dimensions.');
    end

    best_Q = -Inf;
    best_M = [];

    for iteration = 1:n_iterations
        [candidate_M, candidate_Q] = community_louvain( ...
            W, gamma, [], 'modularity');

        if candidate_Q > best_Q
            best_Q = candidate_Q;
            best_M = candidate_M;
        end
    end

    Q_group(subject) = best_Q;
    M_group(subject, :) = best_M(:)';
end


end


function n_modules = count_modules(M_group)

n_subjects = size(M_group, 1);
n_modules = zeros(n_subjects, 1);

for subject = 1:n_subjects
    n_modules(subject) = numel(unique(M_group(subject, :)));
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


function [p_value, W_group_1_observed] = ...
    wilcoxon_rank_sum_permutation(group_1, group_2, group_mask)

group_1 = group_1(:);
group_2 = group_2(:);

if any(~isfinite(group_1)) || any(~isfinite(group_2))
    error('Non-finite values encountered in an unpaired comparison.');
end

n_group_1 = numel(group_1);
n_group_2 = numel(group_2);
pooled = [group_1; group_2];
ranks = tiedrank(pooled);
ranks = ranks(:);

W_group_1_observed = sum(ranks(1:n_group_1));
expected_W = n_group_1 * (n_group_1 + n_group_2 + 1) / 2;
observed_statistic = abs(W_group_1_observed - expected_W);

if size(group_mask, 2) ~= n_group_1 + n_group_2
    error('Permutation group-mask dimensions are inconsistent.');
end

permuted_W = double(group_mask) * ranks;
permuted_statistics = abs(permuted_W - expected_W);

tolerance = 1e-12;
p_value = (sum(permuted_statistics >= ...
    observed_statistic - tolerance) + 1) / (size(group_mask, 1) + 1);


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


function d = cohens_d_independent(group_1, group_2)

group_1 = group_1(:);
group_2 = group_2(:);
n_group_1 = numel(group_1);
n_group_2 = numel(group_2);

pooled_standard_deviation = sqrt( ...
    ((n_group_1 - 1) * var(group_1, 0) + ...
    (n_group_2 - 1) * var(group_2, 0)) / ...
    (n_group_1 + n_group_2 - 2));
mean_difference = mean(group_1) - mean(group_2);

if pooled_standard_deviation == 0
    if mean_difference == 0
        d = 0;
    else
        d = sign(mean_difference) * Inf;
    end
else
    d = mean_difference / pooled_standard_deviation;
end


end


function statistics_table = build_statistics_table( ...
    Q_esc_pre, Q_esc_post, Q_psilo_pre, Q_psilo_post, ...
    modules_esc_pre, modules_esc_post, ...
    modules_psilo_pre, modules_psilo_post, ...
    statistics, effect_sizes, p_values)

delta_Q_esc = Q_esc_post - Q_esc_pre;
delta_Q_psilo = Q_psilo_post - Q_psilo_pre;
delta_modules_esc = modules_esc_post - modules_esc_pre;
delta_modules_psilo = modules_psilo_post - modules_psilo_pre;

group_A = { ...
    Q_esc_post; Q_psilo_post; delta_Q_esc; ...
    modules_esc_post; modules_psilo_post; delta_modules_esc};
group_B = { ...
    Q_esc_pre; Q_psilo_pre; delta_Q_psilo; ...
    modules_esc_pre; modules_psilo_pre; delta_modules_psilo};

measure = [ ...
    "Q"; "Q"; "Q"; ...
    "NumberOfModules"; "NumberOfModules"; "NumberOfModules"];
comparison = [ ...
    "Escitalopram post vs pre"; ...
    "Psilocybin post vs pre"; ...
    "Escitalopram delta vs psilocybin delta"; ...
    "Escitalopram post vs pre"; ...
    "Psilocybin post vs pre"; ...
    "Escitalopram delta vs psilocybin delta"];
test = [ ...
    "Wilcoxon signed-rank"; "Wilcoxon signed-rank"; ...
    "Wilcoxon rank-sum"; "Wilcoxon signed-rank"; ...
    "Wilcoxon signed-rank"; "Wilcoxon rank-sum"];
group_A_label = [ ...
    "Post"; "Post"; "Escitalopram delta"; ...
    "Post"; "Post"; "Escitalopram delta"];
group_B_label = [ ...
    "Pre"; "Pre"; "Psilocybin delta"; ...
    "Pre"; "Pre"; "Psilocybin delta"];
effect_size_type = [ ...
    "Cohen's dz"; "Cohen's dz"; "Pooled Cohen's d"; ...
    "Cohen's dz"; "Cohen's dz"; "Pooled Cohen's d"];

n_A = zeros(6, 1);
n_B = zeros(6, 1);
mean_A = zeros(6, 1);
standard_deviation_A = zeros(6, 1);
mean_B = zeros(6, 1);
standard_deviation_B = zeros(6, 1);

for row = 1:6
    n_A(row) = numel(group_A{row});
    n_B(row) = numel(group_B{row});
    mean_A(row) = mean(group_A{row});
    standard_deviation_A(row) = std(group_A{row});
    mean_B(row) = mean(group_B{row});
    standard_deviation_B(row) = std(group_B{row});
end

mean_A_minus_B = mean_A - mean_B;

statistics_table = table( ...
    measure, comparison, test, group_A_label, group_B_label, ...
    n_A, n_B, mean_A, standard_deviation_A, ...
    mean_B, standard_deviation_B, mean_A_minus_B, ...
    statistics, effect_size_type, effect_sizes, p_values, ...
    'VariableNames', { ...
    'Measure', 'Comparison', 'Test', 'GroupA', 'GroupB', ...
    'N_A', 'N_B', 'MeanA', 'SD_A', 'MeanB', 'SD_B', ...
    'MeanAMinusB', 'WilcoxonStatistic', 'EffectSizeType', ...
    'EffectSize', 'PermutationP'});


end

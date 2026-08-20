function results = regional_gbc_analysis(data_source, ...
    psilocybin_pre_dir, psilocybin_post_dir, ...
    escitalopram_pre_dir, escitalopram_post_dir, ...
    labels_file, output_dir)
% Regional analysis of treatment-related changes in model-derived GBC.

% The function compares pre- and post-treatment GBC within the psilocybin
% and escitalopram groups and compares GBC changes between treatments.
% Regional P values are estimated using 10,000 Wilcoxon permutations and
% corrected separately for each test family using Benjamini-Hochberg FDR.
% GBC is computed as mean(GEC, 1), after replacing NaNs with zero.

% Example:
% results = regional_gbc_analysis('vortex', ...
%     psilocybin_pre_dir, psilocybin_post_dir, ...
%     escitalopram_pre_dir, escitalopram_post_dir, ...
%     'schaefer100_region_labels.xlsx', 'regional_results');

if nargin < 6
    error(['Required inputs: data_source, four model directories, ' ...
        'and labels_file.']);
end

if nargin < 7 || isempty(output_dir)
    output_dir = pwd;
end

data_source = lower(strtrim(char(data_source)));
output_dir = char(output_dir);

switch data_source
    case 'signal'
        file_pattern = 'model_hopf_FC_*.mat';
    case 'vortex'
        file_pattern = 'model_vortex_partition_social_*.mat';
    otherwise
        error('Unknown data_source. Use ''signal'' or ''vortex''.');
end

if exist(output_dir, 'dir') ~= 7
    mkdir(output_dir);
end

variable_name = 'Ceff_task1';
n_permutations = 10000;
fdr_alpha = 0.05;
random_seed = 42;
rng(random_seed, 'twister');

fprintf('\nRegional GBC analysis | %s space\n', upper(data_source));
fprintf('Permutations: %d | BH-FDR alpha: %.3f\n', ...
    n_permutations, fdr_alpha);

[GEC_pre_psilocybin, names_pre_psilocybin] = ...
    load_connectivity_group( ...
        psilocybin_pre_dir, file_pattern, variable_name);
[GEC_post_psilocybin, names_post_psilocybin] = ...
    load_connectivity_group( ...
        psilocybin_post_dir, file_pattern, variable_name);
[GEC_pre_escitalopram, names_pre_escitalopram] = ...
    load_connectivity_group( ...
        escitalopram_pre_dir, file_pattern, variable_name);
[GEC_post_escitalopram, names_post_escitalopram] = ...
    load_connectivity_group( ...
        escitalopram_post_dir, file_pattern, variable_name);

if size(GEC_pre_psilocybin, 1) ~= 22
    error('Expected 22 psilocybin participants; found %d.', ...
        size(GEC_pre_psilocybin, 1));
end

if size(GEC_pre_escitalopram, 1) ~= 20
    error('Expected 20 escitalopram participants; found %d.', ...
        size(GEC_pre_escitalopram, 1));
end

if ~isequal(size(GEC_pre_psilocybin), size(GEC_post_psilocybin))
    error('Psilocybin pre- and post-treatment arrays differ in size.');
end

if ~isequal(size(GEC_pre_escitalopram), size(GEC_post_escitalopram))
    error('Escitalopram pre- and post-treatment arrays differ in size.');
end

if ~isequal(names_pre_psilocybin, names_post_psilocybin)
    error('Psilocybin pre- and post-treatment filenames do not match.');
end

if ~isequal(names_pre_escitalopram, names_post_escitalopram)
    error('Escitalopram pre- and post-treatment filenames do not match.');
end

n_psilocybin = size(GEC_pre_psilocybin, 1);
n_escitalopram = size(GEC_pre_escitalopram, 1);
n_nodes = size(GEC_pre_psilocybin, 2);

if n_nodes ~= 100 || size(GEC_pre_psilocybin, 3) ~= 100
    error('Expected 100 x 100 connectivity matrices.');
end

if size(GEC_pre_escitalopram, 2) ~= n_nodes || ...
        size(GEC_pre_escitalopram, 3) ~= n_nodes
    error('Treatment groups have different matrix dimensions.');
end

[region_labels, yeo_networks] = load_region_labels( ...
    labels_file, n_nodes);
region_index = (1:n_nodes)';

GBC_pre_psilocybin = compute_regional_gbc(GEC_pre_psilocybin);
GBC_post_psilocybin = compute_regional_gbc(GEC_post_psilocybin);
GBC_pre_escitalopram = compute_regional_gbc(GEC_pre_escitalopram);
GBC_post_escitalopram = compute_regional_gbc(GEC_post_escitalopram);

delta_GBC_psilocybin = GBC_post_psilocybin - GBC_pre_psilocybin;
delta_GBC_escitalopram = GBC_post_escitalopram - GBC_pre_escitalopram;

sign_mask_psilocybin = rand( ...
    n_permutations, n_psilocybin) > 0.5;
sign_mask_escitalopram = rand( ...
    n_permutations, n_escitalopram) > 0.5;

n_all = n_psilocybin + n_escitalopram;
group_mask = false(n_permutations, n_all);
for permutation = 1:n_permutations
    selected = randperm(n_all, n_psilocybin);
    group_mask(permutation, selected) = true;
end

p_psilocybin = nan(n_nodes, 1);
p_escitalopram = nan(n_nodes, 1);
p_between = nan(n_nodes, 1);

W_psilocybin = nan(n_nodes, 1);
W_escitalopram = nan(n_nodes, 1);
W_between = nan(n_nodes, 1);

d_psilocybin = nan(n_nodes, 1);
d_escitalopram = nan(n_nodes, 1);
d_between = nan(n_nodes, 1);

for region = 1:n_nodes
    [p_psilocybin(region), W_psilocybin(region)] = ...
        wilcoxon_signed_rank_permutation( ...
            delta_GBC_psilocybin(:, region), ...
            sign_mask_psilocybin);
    d_psilocybin(region) = cohens_dz( ...
        delta_GBC_psilocybin(:, region));

    [p_escitalopram(region), W_escitalopram(region)] = ...
        wilcoxon_signed_rank_permutation( ...
            delta_GBC_escitalopram(:, region), ...
            sign_mask_escitalopram);
    d_escitalopram(region) = cohens_dz( ...
        delta_GBC_escitalopram(:, region));

    [p_between(region), W_between(region)] = ...
        wilcoxon_rank_sum_permutation( ...
            delta_GBC_psilocybin(:, region), ...
            delta_GBC_escitalopram(:, region), group_mask);
    d_between(region) = cohens_d_independent( ...
        delta_GBC_psilocybin(:, region), ...
        delta_GBC_escitalopram(:, region));
end

[q_psilocybin, significant_psilocybin] = ...
    benjamini_hochberg(p_psilocybin, fdr_alpha);
[q_escitalopram, significant_escitalopram] = ...
    benjamini_hochberg(p_escitalopram, fdr_alpha);
[q_between, significant_between] = ...
    benjamini_hochberg(p_between, fdr_alpha);

mean_pre_psilocybin = mean(GBC_pre_psilocybin, 1)';
median_pre_psilocybin = median(GBC_pre_psilocybin, 1)';
mean_post_psilocybin = mean(GBC_post_psilocybin, 1)';
median_post_psilocybin = median(GBC_post_psilocybin, 1)';
mean_delta_psilocybin = mean(delta_GBC_psilocybin, 1)';
median_delta_psilocybin = median(delta_GBC_psilocybin, 1)';

mean_pre_escitalopram = mean(GBC_pre_escitalopram, 1)';
median_pre_escitalopram = median(GBC_pre_escitalopram, 1)';
mean_post_escitalopram = mean(GBC_post_escitalopram, 1)';
median_post_escitalopram = median(GBC_post_escitalopram, 1)';
mean_delta_escitalopram = mean(delta_GBC_escitalopram, 1)';
median_delta_escitalopram = median(delta_GBC_escitalopram, 1)';

mean_delta_difference = ...
    mean_delta_psilocybin - mean_delta_escitalopram;
median_delta_difference = ...
    median_delta_psilocybin - median_delta_escitalopram;

T_psilocybin = table( ...
    region_index, region_labels, yeo_networks, ...
    mean_pre_psilocybin, median_pre_psilocybin, ...
    mean_post_psilocybin, median_post_psilocybin, ...
    mean_delta_psilocybin, median_delta_psilocybin, ...
    d_psilocybin, W_psilocybin, p_psilocybin, ...
    q_psilocybin, significant_psilocybin, ...
    'VariableNames', { ...
    'RegionIndex', 'RegionLabel', 'YeoNetwork', ...
    'MeanGBC_Pre', 'MedianGBC_Pre', ...
    'MeanGBC_Post', 'MedianGBC_Post', ...
    'MeanDeltaGBC', 'MedianDeltaGBC', ...
    'Cohens_dz', 'WilcoxonSignedRank_Wplus', ...
    'PermutationP', 'FDR_q', 'FDR_q_lt_0_05'});

T_escitalopram = table( ...
    region_index, region_labels, yeo_networks, ...
    mean_pre_escitalopram, median_pre_escitalopram, ...
    mean_post_escitalopram, median_post_escitalopram, ...
    mean_delta_escitalopram, median_delta_escitalopram, ...
    d_escitalopram, W_escitalopram, p_escitalopram, ...
    q_escitalopram, significant_escitalopram, ...
    'VariableNames', { ...
    'RegionIndex', 'RegionLabel', 'YeoNetwork', ...
    'MeanGBC_Pre', 'MedianGBC_Pre', ...
    'MeanGBC_Post', 'MedianGBC_Post', ...
    'MeanDeltaGBC', 'MedianDeltaGBC', ...
    'Cohens_dz', 'WilcoxonSignedRank_Wplus', ...
    'PermutationP', 'FDR_q', 'FDR_q_lt_0_05'});

T_between = table( ...
    region_index, region_labels, yeo_networks, ...
    mean_delta_psilocybin, median_delta_psilocybin, ...
    mean_delta_escitalopram, median_delta_escitalopram, ...
    mean_delta_difference, median_delta_difference, ...
    d_between, W_between, p_between, q_between, significant_between, ...
    'VariableNames', { ...
    'RegionIndex', 'RegionLabel', 'YeoNetwork', ...
    'MeanDeltaGBC_Psilocybin', 'MedianDeltaGBC_Psilocybin', ...
    'MeanDeltaGBC_Escitalopram', 'MedianDeltaGBC_Escitalopram', ...
    'MeanDeltaDifference_PsiloMinusEsc', ...
    'MedianDeltaDifference_PsiloMinusEsc', ...
    'Cohens_d', 'WilcoxonRankSum_W_Psilocybin', ...
    'PermutationP', 'FDR_q', 'FDR_q_lt_0_05'});

n_top = ceil(0.20 * n_nodes);
[~, order_psilocybin] = sort(abs(mean_delta_psilocybin), 'descend');
[~, order_escitalopram] = sort(abs(mean_delta_escitalopram), 'descend');

T_top_psilocybin = T_psilocybin(order_psilocybin(1:n_top), :);
T_top_psilocybin.AbsMeanDeltaGBC = ...
    abs(T_top_psilocybin.MeanDeltaGBC);
T_top_psilocybin.RankByAbsMeanDelta = (1:n_top)';
T_top_psilocybin = movevars( ...
    T_top_psilocybin, ...
    {'RankByAbsMeanDelta', 'AbsMeanDeltaGBC'}, ...
    'After', 'YeoNetwork');

T_top_escitalopram = T_escitalopram( ...
    order_escitalopram(1:n_top), :);
T_top_escitalopram.AbsMeanDeltaGBC = ...
    abs(T_top_escitalopram.MeanDeltaGBC);
T_top_escitalopram.RankByAbsMeanDelta = (1:n_top)';
T_top_escitalopram = movevars( ...
    T_top_escitalopram, ...
    {'RankByAbsMeanDelta', 'AbsMeanDeltaGBC'}, ...
    'After', 'YeoNetwork');

T_network_psilocybin = network_counts( ...
    yeo_networks(significant_psilocybin));
T_network_escitalopram = network_counts( ...
    yeo_networks(significant_escitalopram));
T_network_between = network_counts( ...
    yeo_networks(significant_between));

n_extreme = 5;
T_extreme_psilocybin = signed_extremes_table( ...
    T_psilocybin, 'MeanDeltaGBC', n_extreme);
T_extreme_escitalopram = signed_extremes_table( ...
    T_escitalopram, 'MeanDeltaGBC', n_extreme);
T_extreme_between = signed_extremes_table( ...
    T_between, 'MeanDeltaDifference_PsiloMinusEsc', n_extreme);

excel_file = fullfile( ...
    output_dir, sprintf('regional_GBC_analysis_%s.xlsx', data_source));
if exist(excel_file, 'file') == 2
    delete(excel_file);
end

writetable(T_psilocybin, excel_file, 'Sheet', 'Psilocybin');
writetable(T_escitalopram, excel_file, 'Sheet', 'Escitalopram');
writetable(T_between, excel_file, 'Sheet', 'Between_treatments');
writetable(T_top_psilocybin, excel_file, 'Sheet', 'Top20_psilocybin');
writetable(T_top_escitalopram, excel_file, 'Sheet', 'Top20_escitalopram');
writetable(T_network_psilocybin, excel_file, ...
    'Sheet', 'Networks_psilocybin');
writetable(T_network_escitalopram, excel_file, ...
    'Sheet', 'Networks_escitalopram');
writetable(T_network_between, excel_file, 'Sheet', 'Networks_between');
writetable(T_extreme_psilocybin, excel_file, ...
    'Sheet', 'Extremes_psilocybin');
writetable(T_extreme_escitalopram, excel_file, ...
    'Sheet', 'Extremes_escitalopram');
writetable(T_extreme_between, excel_file, 'Sheet', 'Extremes_between');

results = struct();
results.settings.data_source = data_source;
results.settings.n_permutations = n_permutations;
results.settings.fdr_alpha = fdr_alpha;
results.settings.random_seed = random_seed;
results.settings.gbc_definition = ...
    'NaNs replaced with zero; mean(GEC,1); diagonal retained';
results.settings.delta_definition = 'post minus pre';
results.settings.paired_effect_size = ...
    'Cohen''s dz = mean(post-pre) / SD(post-pre)';
results.settings.between_effect_size = ...
    'Pooled-SD Cohen''s d, psilocybin minus escitalopram';

results.labels.RegionIndex = region_index;
results.labels.RegionLabel = region_labels;
results.labels.YeoNetwork = yeo_networks;

results.psilocybin.filenames_pre = names_pre_psilocybin;
results.psilocybin.filenames_post = names_post_psilocybin;
results.psilocybin.GBC_pre = GBC_pre_psilocybin;
results.psilocybin.GBC_post = GBC_post_psilocybin;
results.psilocybin.delta_GBC = delta_GBC_psilocybin;
results.psilocybin.table = T_psilocybin;
results.psilocybin.top_20_percent = T_top_psilocybin;
results.psilocybin.network_counts = T_network_psilocybin;
results.psilocybin.extreme_changes = T_extreme_psilocybin;

results.escitalopram.filenames_pre = names_pre_escitalopram;
results.escitalopram.filenames_post = names_post_escitalopram;
results.escitalopram.GBC_pre = GBC_pre_escitalopram;
results.escitalopram.GBC_post = GBC_post_escitalopram;
results.escitalopram.delta_GBC = delta_GBC_escitalopram;
results.escitalopram.table = T_escitalopram;
results.escitalopram.top_20_percent = T_top_escitalopram;
results.escitalopram.network_counts = T_network_escitalopram;
results.escitalopram.extreme_changes = T_extreme_escitalopram;

results.between.table = T_between;
results.between.network_counts = T_network_between;
results.between.extreme_changes = T_extreme_between;

mat_file = fullfile( ...
    output_dir, sprintf( ...
        'regional_GBC_analysis_results_%s.mat', data_source));
save(mat_file, 'results', '-v7.3');

fprintf('\nFDR-significant regions (q < %.2f):\n', fdr_alpha);
fprintf('Psilocybin: %d / %d\n', ...
    sum(significant_psilocybin), n_nodes);
fprintf('Escitalopram: %d / %d\n', ...
    sum(significant_escitalopram), n_nodes);
fprintf('Between-treatment delta: %d / %d\n', ...
    sum(significant_between), n_nodes);
fprintf('Saved results to %s\n', output_dir);

end


function [matrices, sorted_names] = load_connectivity_group( ...
    directory, file_pattern, variable_name)

directory = char(directory);
if exist(directory, 'dir') ~= 7
    error('Connectivity directory does not exist: %s', directory);
end

files = dir(fullfile(directory, file_pattern));
if isempty(files)
    error('No files found in %s with pattern %s.', ...
        directory, file_pattern);
end

[~, order] = sort({files.name});
files = files(order);
sorted_names = string({files.name})';

matrices = [];
for subject = 1:numel(files)
    file_path = fullfile(files(subject).folder, files(subject).name);
    loaded = load(file_path, variable_name);

    if ~isfield(loaded, variable_name)
        error('Variable %s was not found in %s.', ...
            variable_name, file_path);
    end

    matrix = double(loaded.(variable_name));
    if ndims(matrix) ~= 2 || size(matrix, 1) ~= size(matrix, 2)
        error('Connectivity matrix in %s must be square.', file_path);
    end
    if any(isinf(matrix(:)))
        error('Infinite connectivity values found in %s.', file_path);
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


function [region_labels, yeo_networks] = load_region_labels( ...
    labels_file, n_nodes)

labels_file = char(labels_file);
if exist(labels_file, 'file') ~= 2
    error('Region-label file does not exist: %s', labels_file);
end

label_table = readtable(labels_file);
required_columns = {'RegionIndex', 'RegionLabel', 'YeoNetwork'};

if ~all(ismember(required_columns, label_table.Properties.VariableNames))
    error(['The label file must contain RegionIndex, RegionLabel, ' ...
        'and YeoNetwork columns.']);
end

if height(label_table) ~= n_nodes
    error('Expected %d label rows; found %d.', ...
        n_nodes, height(label_table));
end

if ~isequal(label_table.RegionIndex(:), (1:n_nodes)')
    error('RegionIndex must run consecutively from 1 to %d.', n_nodes);
end

region_labels = string(label_table.RegionLabel);
yeo_networks = string(label_table.YeoNetwork);

if any(strlength(region_labels) == 0) || ...
        any(strlength(yeo_networks) == 0)
    error('At least one region or Yeo-network label is missing.');
end


end


function GBC = compute_regional_gbc(matrices)

n_nodes = size(matrices, 2);
if size(matrices, 3) ~= n_nodes
    error('Connectivity matrices must be square.');
end

matrices(isnan(matrices)) = 0;
GBC = squeeze(mean(matrices, 2));

if isvector(GBC)
    GBC = reshape(GBC, 1, []);
end


end


function [p_value, W_plus_observed] = ...
    wilcoxon_signed_rank_permutation(delta, sign_mask)

delta = delta(:);
if any(~isfinite(delta))
    error('Non-finite values encountered in a paired comparison.');
end

nonzero = delta ~= 0;
differences = delta(nonzero);

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


function d = cohens_dz(delta)

delta = delta(:);
mean_difference = mean(delta);
standard_deviation = std(delta, 0);

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


function [q_values, significant] = benjamini_hochberg(p_values, alpha)

p_values = p_values(:);
q_values = nan(size(p_values));
valid = isfinite(p_values);
valid_p = p_values(valid);
n_tests = numel(valid_p);

if n_tests == 0
    significant = false(size(p_values));
    return
end

[sorted_p, order] = sort(valid_p, 'ascend');
sorted_q = sorted_p .* n_tests ./ (1:n_tests)';
sorted_q = flipud(cummin(flipud(sorted_q)));
sorted_q = min(sorted_q, 1);

valid_q = nan(n_tests, 1);
valid_q(order) = sorted_q;
q_values(valid) = valid_q;
significant = q_values < alpha;


end


function table_out = network_counts(networks)

networks = string(networks(:));
if isempty(networks)
    table_out = table( ...
        strings(0, 1), zeros(0, 1), ...
        'VariableNames', {'YeoNetwork', 'NRegions'});
    return
end


[unique_networks, ~, indices] = unique(networks, 'stable');
counts = accumarray(indices, 1);
table_out = table( ...
    unique_networks, counts, ...
    'VariableNames', {'YeoNetwork', 'NRegions'});
table_out = sortrows(table_out, 'NRegions', 'descend');


end


function table_out = signed_extremes_table( ...
    table_in, value_variable, n_extreme)

values = table_in.(value_variable);
positive_candidates = find(values > 0);
negative_candidates = find(values < 0);

[~, positive_order] = sort( ...
    values(positive_candidates), 'descend');
[~, negative_order] = sort( ...
    values(negative_candidates), 'ascend');

positive_indices = positive_candidates(positive_order( ...
    1:min(n_extreme, numel(positive_order))));
negative_indices = negative_candidates(negative_order( ...
    1:min(n_extreme, numel(negative_order))));

positive_table = table_in(positive_indices, :);
negative_table = table_in(negative_indices, :);
positive_table.Direction = repmat( ...
    "Positive", height(positive_table), 1);
negative_table.Direction = repmat( ...
    "Negative", height(negative_table), 1);

table_out = [positive_table; negative_table];
table_out = movevars(table_out, 'Direction', 'Before', 1);


end

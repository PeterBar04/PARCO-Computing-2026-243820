function [data_time, data_cache] = read_results()
% Read the CSV files made by the benchmark of the SpMV

base_folder = fileparts(mfilename('fullpath'));

results_folder = fullfile(base_folder, '..', 'results');

time_file_struct = dir(fullfile(results_folder, '*time_results.csv'));
    if isempty(time_file_struct)
        error('❌  "*time.csv" not found in: %s', results_folder);
    end
    time_filename = fullfile(results_folder, time_file_struct(1).name);

    % Cerca il file cache
    cache_file_struct = dir(fullfile(results_folder, '*cache_results.csv'));
    if isempty(cache_file_struct)
        error('❌ "*cache.csv" not found in: %s', results_folder);
    end
    cache_filename = fullfile(results_folder, cache_file_struct(1).name);

data_cache=read_results_cache(cache_filename);   
data_time=read_results_time(time_filename);

fprintf('✅ Files "%s and %s" read correctly.\n', time_filename, cache_filename);
    
end
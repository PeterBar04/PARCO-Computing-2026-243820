function data_by_matrix = read_results_time(filename)

%   data_by_matrix = read_results('results_time.csv');

    if nargin < 1
        filename = 'results.csv';
    end


    opts = detectImportOptions(filename, 'Delimiter', ',', 'VariableNamingRule', 'preserve');
    opts.VariableNamesLine = 1;
    opts.DataLines = 2;
    opts = setvartype(opts, {'Matrix', 'Mode', 'Schedule'}, 'string');
    opts = setvartype(opts, {'Threads','Chunk_size','p90_run_time','p90_speed_up','p90_efficiency'}, 'double');


    data = readtable(filename, opts);

    % Rimuove eventuali spazi o virgolette
    data.Matrix = strrep(data.Matrix, '"', '');
    data.Schedule = strrep(data.Schedule, '"', '');
    data.Mode = strrep(data.Mode, '"', '');
    data.Schedule = strtrim(data.Schedule);


    matrices = unique(data.Matrix);
    data_by_matrix = struct();

    for i = 1:numel(matrices)
        mname = matrices(i);
        tbl = data(data.Matrix == mname, :);

        % Nome campo senza punti (MATLAB non li accetta)
        fieldname = matlab.lang.makeValidName(strrep(mname, '.', '_'));

        data_by_matrix.(fieldname) = tbl;
    end


    fprintf('✅ File "%s" correctly read.\n', filename);
    fprintf('Contains %d matrixes:\n', numel(matrices));
    disp(matrices);

end
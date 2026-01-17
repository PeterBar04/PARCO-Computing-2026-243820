function plot_results()

    [data_time_by_matrix, data_cache_by_matrix] = read_results();

    matrices = fieldnames(data_time_by_matrix);
    chunk_sizes = [10, 100, 1000];
    colors = lines(numel(chunk_sizes));

    for m = 1:numel(matrices)

        matrix_name = matrices{m};

        % time data
        T_time = data_time_by_matrix.(matrix_name);

        % cache data
        if ~isfield(data_cache_by_matrix, matrix_name)
            warning("⚠ Cache data missing for matrix %s — skipping", matrix_name);
            continue;
        end
        T_cache = data_cache_by_matrix.(matrix_name);

        % Filter Parallel-only
        if ismember("Mode", T_time.Properties.VariableNames)
            T_time = T_time(T_time.Mode == "Parallel", :);
        end
        if ismember("Mode", T_cache.Properties.VariableNames)
            T_cache = T_cache(T_cache.Mode == "Parallel", :);
        end

        schedules = intersect(unique(T_time.Schedule), unique(T_cache.Schedule));
        threads   = sort(unique(T_time.Threads(~isnan(T_time.Threads))));

        for s = 1:numel(schedules)
            sched = schedules(s);

            fprintf("\n📊 %s — Schedule %s\n", matrix_name, sched);

            %% Allocate matrices

            Nthr = length(threads);
            Nchunk = length(chunk_sizes);

            times        = nan(Nthr, Nchunk);
            speedup      = nan(Nthr, Nchunk);
            efficiency   = nan(Nthr, Nchunk);
            l1_miss      = nan(Nthr, Nchunk);
            ll_miss      = nan(Nthr, Nchunk);

            %% Fill TIME data
            for i = 1:Nthr
                th = threads(i);
                for j = 1:Nchunk
                    cs = chunk_sizes(j);

                    id = T_time.Threads == th & ...
                         T_time.Schedule == sched & ...
                         T_time.Chunk_size == cs;

                    if any(id)
                        times(i,j)      = mean(T_time.p90_run_time(id));
                        speedup(i,j)    = mean(T_time.p90_speed_up(id));
                        efficiency(i,j) = mean(T_time.p90_efficiency(id));
                    end
                end
            end

            %% Fill CACHE data
            for i = 1:Nthr
                th = threads(i);
                for j = 1:Nchunk
                    cs = chunk_sizes(j);

                    id = T_cache.Threads == th & ...
                         T_cache.Schedule == sched & ...
                         T_cache.Chunk_size == cs;

                    if any(id)
                        if ismember("l1_miss_rate", T_cache.Properties.VariableNames)
                            l1_miss(i,j) = mean(T_cache.l1_miss_rate(id));
                        end
                        if ismember("ll_miss_rate", T_cache.Properties.VariableNames)
                            ll_miss(i,j) = mean(T_cache.ll_miss_rate(id));
                        end
                    end
                end
            end

            %% Skip if totally empty
            if all(isnan(times),"all") && all(isnan(l1_miss),"all")
                fprintf("⚠  No data for %s (%s)\n", matrix_name, sched);
                continue;
            end

            %% === FIGURE ===
            figure("Name", sprintf("%s - %s", matrix_name, sched), ...
                   "Position", [100 100 900 1200]);

            tlo = tiledlayout(2,2,'TileSpacing','compact');

            %% 1) Speed-up
            nexttile; hold on;
            for j = 1:Nchunk
                plot(threads, speedup(:,j), "-o", "Color", colors(j,:), ...
                     "DisplayName", sprintf("chunk=%d", chunk_sizes(j)));
            end
            hold off; grid on;
            ylabel("Speed-up"); xlabel("Threads");
            title("Speed-up");
            legend("Location","northoutside","Orientation","horizontal");

            %% 2) Efficiency (heatmap)
            nexttile;
            try
                heatmap(string(threads), string(chunk_sizes), efficiency', ...
                        "Colormap", parula, "CellLabelColor","none");
                ylabel("Chunk size"); xlabel("Threads");
                title("Efficiency (%)");
            catch
                hold on;
                for j = 1:Nchunk
                    plot(threads, efficiency(:,j), "-o", "Color", colors(j,:), ...
                         "DisplayName", sprintf("chunk=%d", chunk_sizes(j)));
                end
                hold off; grid on;
                ylabel("Efficiency (%)"); xlabel("Threads");
                title("Efficiency");
            end

            %% 3) L1 miss rate
            nexttile; hold on;
            for j = 1:Nchunk
                plot(threads, l1_miss(:,j), "-o", "Color", colors(j,:), ...
                     "DisplayName", sprintf("chunk=%d", chunk_sizes(j)));
            end
            hold off; grid on;
            ylabel("L1 miss rate"); xlabel("Threads");
            title("L1 Miss Rate");

            %% 4) LL miss rate
            nexttile; hold on;
            for j = 1:Nchunk
                plot(threads, ll_miss(:,j), "-o", "Color", colors(j,:), ...
                     "DisplayName", sprintf("chunk=%d", chunk_sizes(j)));
            end
            hold off; grid on;
            ylabel("LL miss rate"); xlabel("Threads");
            title("Last-level Miss Rate");

            sgtitle(sprintf("%s — %s", matrix_name, sched), ...
                    "Interpreter","none","FontWeight","bold");

            saveas(gcf, fullfile('..', 'plots', sprintf("%s_%s.png", matrix_name, sched)));
            close(gcf);
        end
    end
end

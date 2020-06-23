function [ B, E ] = loadmtx( file, T, N )
%LOADMTX writed by lanlan 
% load the file .mtx saved by python's pakage scipy.sparse
%   此处显示详细说明
    A = zeros(T, N*N);
    B = zeros(N, N, T);  % adjacency matrix for dynamic network
    E = zeros(N, T);  % E(i, t) is 1 if node i exists in t snapshot
    fout = fopen(file, 'r');
    flag = 0;
    while feof(fout) == 0;
        if flag < 3;
            tline = fgetl(fout);
            flag = flag + 1;
            continue;
        end;
        tline = fgetl(fout);
        str = deblank(tline);
        S = regexp(str, '\s+', 'split');
        A(str2num(S{1}), str2num(S{2})) = str2num(S{3});
    end;
    for t=1:T
        a = reshape(A(t,:), N, []);
        a = a + a';
        exist_node = find(sum(a, 2));  % index of existing nodes
        E(exist_node,t) = 1;
        B(:,:,t) = a;
    end
end


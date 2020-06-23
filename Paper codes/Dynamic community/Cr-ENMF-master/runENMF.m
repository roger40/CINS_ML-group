function [Z]=runENMF(dataset)
% --Input  
%   --dataset is index of dataset
%     0:R0  1:R4-4  2:R4-8  3:R3-4  4:R3-8  5:SBM
%   author: Lanlan Yu
% --parameter
%   --B is a [n,k,T] matrix
%   --H is a [k,k,T] matrix 
%   --F is a [k,n,T] matrix 
% --Output
%   --Z = [T,N] matrix
    switch dataset
        case 0
%             dataset = 'R0'
            file = 'E:\DATASet\Reddit\reddit\2010-09(sampling ratio 1.0)\R0-\undirected\reddit0.mtx';
            N = 18; 
            T = 4;
            k = 2;
        case 1
%             dataset = 'R4'
            file = 'E:\DATASet\Reddit\reddit\2010-09(sampling ratio 1.0)\R4-\undirected\reddit4.mtx';
            N = 282; 
            T = 4;
            k = 3;
        case 2
%             dataset = 'R4-2'
            file = 'E:\DATASet\Reddit\reddit\2010-09-10(sampling ratio 1.0)\R4-\undirected\reddit8.mtx';
            N = 470; 
            T = 8;
            k = 3;
        case 3
%             dataset = 'R3'
            file = 'E:\DATASet\Reddit\reddit\2010-09(sampling ratio 1.0)\R3-\undirected\reddit3.mtx';
            N = 275; 
            T = 4;
            k = 17;
        case 4
%             dataset = 'R3-2'
            file = 'E:\DATASet\Reddit\reddit\2010-09-10(sampling ratio 1.0)\R3-\undirected\reddit8.mtx';
            N = 339; 
            T = 8;
            k = 17;
        case 5
            file = 'G:\CodeSet\workspace\HGCN\sinmulateFordraw\SBM\sbm_1000.mtx';
            T = 4;
            N = 1000;
            k = 4;
    end
    [path,name,suffix]=fileparts(file);
    folder = fullfile(path, 'community', 'ENMF');
    if ~exist(folder,'dir')
        mkdir(folder);
    end
    
    [W, E] = loadmtx(file, T, N);

%     [B,H,F]=NNMF(W,Iter,k,beta,gamma);
    % F is a [k,n,T] matrix 
% 	for t=1:T
%          Z(:,t) = max(F(:, :, t),[],1);
%     end
% 	Z = Z';

    for beta=0.1:0.1:1
        for gama=0.1:0.1:1
            name = [['beta',num2str(beta)],['gama',num2str(gama)]];
            [B,H,F]=CrNMF(W, k, beta, gama, 2);
            Z = zeros(N, T);
%             disp(F)
            for t=1:T
                exist_nodes = find(E(:,t));
                [a, b] = max(F(:, :, t),[],1);
                Z(exist_nodes,t) = b(exist_nodes);  % zeros is not existing
            end
            Z = trace(Z');
            save([folder,'\',name,'.mat'], 'Z');
%             break
        end
%         break
    end
%     trace
end
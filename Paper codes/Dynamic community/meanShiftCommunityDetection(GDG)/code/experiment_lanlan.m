for dataset=0:1:5
    switch dataset
        case 0
            dataset = 'R0';
            file = 'E:\DATASet\Reddit\reddit\2010-09(sampling ratio 1.0)\R0-\undirected\reddit0.mtx';
            N = 18; 
            T = 4;
            k = 2;
        case 1
            dataset = 'R4';
            file = 'E:\DATASet\Reddit\reddit\2010-09(sampling ratio 1.0)\R4-\undirected\reddit4.mtx';
            N = 282;
            T = 4;
            k = 3;
        case 2
            dataset = 'R4-2';
            file = 'E:\DATASet\Reddit\reddit\2010-09-10(sampling ratio 1.0)\R4-\undirected\reddit8.mtx';
            N = 470; 
            T = 8;
            k = 3;
        case 3
            dataset = 'R3';
            file = 'E:\DATASet\Reddit\reddit\2010-09(sampling ratio 1.0)\R3-\undirected\reddit3.mtx';
            N = 275; 
            T = 4;
            k = 17;
        case 4
            dataset = 'R3-2';
            file = 'E:\DATASet\Reddit\reddit\2010-09-10(sampling ratio 1.0)\R3-\undirected\reddit8.mtx';
            N = 339; 
            T = 8;
            k = 17;
        case 5
            dataset = 'SBM1000';
            file = 'G:\CodeSet\workspace\HGCN\sinmulateFordraw\SBM\sbm_1000.mtx';
            T = 4;
            N = 1000;
            k = 4;
    end
    display(dataset)
    [path,name,suffix]=fileparts(file);
    folder = fullfile(path, 'community', 'GDG');
    if ~exist(folder,'dir')
        mkdir(folder);
    end
    
    % read dataset,A is N*N*T, Temp is the signal of existing nodes
    [A, Temp]  = loadmtx(file, T, N);
    
    for bw=0.1:0.1:1
        K=k;
        name = ['bw',num2str(bw)];
        Z = zeros(N, T);
        for t=1:T
%             display(t);
            Adj = A(:,:,t);
            nodes = find(Temp(:, t));
            
            non_nodes = find(Temp(:, t) == 0);
            Adj(non_nodes, :) = [];
            Adj(:, non_nodes) = [];
%             h=view(biograph(sparse(Adj),[],'ShowWeights','on'))%显示图的结构
%             break
            E=sum(sum(Adj))/2;
%             display(E)
            
            if E == 0
                continue;
            end
            
            [S, ci] = graphconncomp(sparse(Adj));
%             break
            label = intersect(ci,ci);
            signal = 0;
            for i=label
                connected_nodes = find(ci==i);
%                 if i==4
%                 display(length(connected_nodes))
%                 end
%                 错误使用 kmeans (line 269)
%                 X must have more rows than the number of clusters.
                l = length(connected_nodes);
%                 if 1 < l && l <= K
                if l <= K
%                     display('error-------------------------')
                    signal = signal + 1;
                    Z(nodes(connected_nodes),t) = signal;
%                     signal = signal + 1;
                    continue;
%                 elseif 1 == l
%                         continue
                end
                mat = Adj(connected_nodes,connected_nodes);  
                
%                 h=view(biograph(sparse(mat),[],'ShowWeights','on'))%显示图的结构
%                 break
                
                N=length(mat);
                E=sum(sum(mat))/2;
                
                feat=graphallshortestpaths(sparse(mat),'DIRECTED',false); % if you have Bioinformatics Toolbox
%                 feat=allspath(mat); % if you don't have bioinformatics toolbox
% continue
                dPCnt = SimpleMSv1(feat,bw);

                idx = kmeans(dPCnt, K,'MaxIter',10,'Replicates',1,'EmptyAction','singleton');
%                display(idx);
                signal = signal + 1;
                Z(nodes(connected_nodes),t) = idx + signal;
%                 signal = signal + 1;
            end
        end
        ZZ = Z';
        [Z] = trace(Z');
        save([folder,'\',name,'.mat'], 'Z');
    end
%     clear all;
end


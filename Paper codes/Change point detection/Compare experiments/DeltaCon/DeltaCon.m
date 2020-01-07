% change deltadistance by lanlan
function [deltadistance, deltacon, deltaconSTD, t, t_STD] = ...
    DeltaCon( datasetName, ...
    naiveORfast, ...
    edgeFile1, edgeFile2, ...
    percent_init, ...
    nodes )
% tic toc 记录时间：tic记录起始时间，toc 记录时间差，t, t_STD是计�?deltacon, deltaconSTD �?��的时�?%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DeltaCon:                                                               %
%  For the two given graphs, G_A & G_B, with the same number of nodes and %
% different edge sets, run FaBP x times by starting from different nodes. %
% At the end of all the runs, for each graph, stack the final beliefs per %
% run in a matrix, say S_A and S_B respectively. Compute the matusita     %
% distance, d, between the matrices S_A and S_B. The DeltaCon similarity  %
% between the input graphs is given by s = 1 / (1+d), where s \in [0,1],  %
% with s = 0 meaning totally different graphs, and s = 1 meaning          %
% identical graphs.                                                       %
%                                                                         %
% INPUTS:                                                                 %
% 1. datasetName: 'matFiles' if the input graphs are stored in .mat files %
%                 o/w (random) if the input graphs are edge files (txt)   %
% 2. edge file 1    *Requirement*: node numbering starts from 1!          %
% 3. edge file 2                                                          %
% 4. naiveORfast:                                                         %
%           'naive' -> DeltaCon0, the naive, exact algorithm is executed. %
%           'fast'  -> DeltaCon, the fast algorithm is executed.          %
% 5. percent_init: % of nodes in each group for the 'fast' algorithm.     %
%             Default value: 10%.                                         %
% 6. [nodes]: number of nodes in the input graphs                         %
%             If not provided, we set the number of nodes to be the max   %
%             of size of the adjacency matrices A1 and A2.                %
%                                                                         %
%       ======================== FaBP ==========================          %
% BP is approximated by the following linear system (FaBP):               %
% [I + ah*D - ch*A] b_h = phi_h                                           %
%                                                                         %
% ABOUT-HALF APPROXIMATIONS                                               %
% h_h = h - 1/2: this holds for the beliefs, the priors, the messages     %
% and the probabilities. So, all the quantities are in                    %
% "0 (+ or -) epsilon" for small epsilon                                  %
%                                                                         %
% ~~~ NOTES ~~~                                                           %
% ** h_h is selected based on norm l2 (starting point)                    %
% ** We solve the linear system by using the power method                 %
%                                                                         %
%                                                                         %
% CITATION: Danai Koutra, Tai-You Ke, U. Kang, Duen Horng Chau, Hsing-Kuo %
%  Kenneth Pao, Christos Faloutsos:                                       %
%  Unifying Guilt-by-Association Approaches: Theorems and Fast Algorithms.%
%  ECML/PKDD (2) 2011: 245-260                                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin < 4
    disp('>>>> call: sparse_read_data_v2( datasetName, naiveORfast, edgeFile1, edgeFile2, percent_init, nodeCnt(opt) )');
else
    
    %% Initialization
    deltacon = 0;
    deltadistance = 0;   % deltadistance by lanlan
    deltaconSTD = 0;
    t = 0;
    t_STD = 0;
    
    %% CONSTANTS
    % prior belief for the node(s) we are initializing in each run
    % == does not really affect the result for DeltaCon measure ==
    p = 0.51;
    
    %% Load the edge files and create the corresponding adjacency matrices
    if strcmpi( datasetName, 'matFiles')
        A1 = edgeFile1;
        A2 = edgeFile2;
    else
        A1  = edgefile2matrix( edgeFile1 );
        A2 = edgefile2matrix( edgeFile2 );
    end
    
    if nargin < 6
        nodeCnt = max( size(A1,1), size(A2,2) );
    elseif nargin == 6
        nodeCnt = nodes;
    end
    A1(nodeCnt, nodeCnt) = 0;
    A2(nodeCnt, nodeCnt) = 0;
    % Ignore the weights of the edges, in case the input graphs are
    % weighted.
    A1(find(A1)) = 1;
    A2(find(A2)) = 1;
    
    %% Solve the linear system: inv corresponds to [I + ah*D - ch*A]^-1
    
    %% 'fast': we run DeltaCon - so we create k groups of nodes with
    %         %percent of the nodes each.
    if ( strcmpi( naiveORfast, 'fast') )
        if nargin == 5
            percent = percent_init;
        else
            percent = 0.1;
        end
        % Number of node groups to be initialized.
        groups = ceil(1/percent);
        % Number of repetitions of the random algorithm.
        % We report the avg/std of the runtime and similarity score.
        repetitions = 10;
        t_all = zeros(1, repetitions);
        sim = zeros(1, repetitions);
        for j = 1 : repetitions
            
            priors = initPriorsPercent(percent, nodeCnt);   % have random operation
            
            tic
            inv1 = inverse_LBP_w_groupSeed( A1, nodeCnt, priors, groups ) .* (p-0.5);
            tp(1) = toc;
            
            tic
            inv2 = inverse_LBP_w_groupSeed( A2, nodeCnt, priors, groups ) .* (p-0.5);
            tp(2) = toc;
            d(j) = sqrt( sum(sum( ( sqrt(inv1) - sqrt(inv2) ).^2 ) ));   % by lanlan 
            tic
            sim(j) = 1 / (1 + sqrt( sum(sum( ( sqrt(inv1) - sqrt(inv2) ).^2 ) )) );
            tp(3) = toc;
            
            tp(4) = sum(tp(1:3));
            % total time for the jth run
            t_all(j) = tp(4);
        end
        t = mean( t_all(1:j) );
        t_STD = std( t_all(1:j) );
        deltacon = mean( sim(1:j) );
        deltadistance = mean(d(1:j));
        deltaconSTD = std( sim(1:j) );
        %% 'naive': we run FaBP by naively initializing every node in the graphs
    elseif ( strcmpi( naiveORfast, 'naive') )
        
        tic
        inv1 = inverse_LBP( A1, nodeCnt ) .* (p-0.5);
        tp(1) = toc;
        
        tic
        inv2 = inverse_LBP( A2, nodeCnt ) .* (p-0.5);
        tp(2) = toc;
        
        % Computing the DeltaCon similarity
        tic
        deltacon = 1 / (1 + sqrt( sum(sum( (sqrt(inv1) - sqrt(inv2)).^2 ) )) );
        deltadistance = sqrt( sum(sum( (sqrt(inv1) - sqrt(inv2)).^2 ) ));
        tp(3) = toc;
        
        tp(4) = sum(tp(1:3));
        t = tp(4);
        deltaconSTD = 0;
        t_STD = 0;
    end
    
end

end

% Computing the modularity for a given module/commnunity break-down
% Defined as: Q=sum_over_modules_i (eii-ai^2) (eq 5) in Newman and Girvan.
% eij = fraction of edges that connect community i to community j, ai=sum_j (eij)
% Source: Newman, M.E.J., Girvan, M., "Finding and evaluating community structure in networks"
% Also: "Fast algorithm for detecting community structure in networks", Mark Newman
% Inputs: adjacency matrix and set modules as cell array of vectors, ex: {[1,2,3],[4,5,6]}
% Outputs: modularity metric, in [-1,1]
% Other functions used: numedges.m
% Last updated: June 13, 2011

function Q=modularityMIT(idx,adj)
NComm=max(idx);

nedges=numedges(adj); % total number of edges

Q = 0;
for m=1:NComm
   modules=find(idx==m);
  e_mm=numedges(adj(modules,modules))/nedges;
  a_m=sum(sum(adj(modules,:)))/(2*nedges);
  Q = Q + (e_mm - a_m^2);
  
end
end

% $$$ % alternative: Q = sum_ij { 1/2m [Aij-kikj/2m]delta(ci,cj) } = 
% $$$ % = sum_ij Aij/2m delta(ci,cj) - sum_ij kikj/4m^2 delta(ci,cj) = 
% $$$ % = sum_modules e_mm - sum_modules (kikj/4m^2) =
% $$$ % = sum_modules (e_mm - ((sum_i ki)/2m)^2)
% $$$ 
% $$$ n = numnodes(adj);
% $$$ m = numedges(adj);
% $$$ 
% $$$ mod={};
% $$$ for mm=1:length(modules)
% $$$   for ii=1:length(modules{mm})
% $$$     mod{modules{mm}(ii)}=modules{mm};
% $$$   end
% $$$ end
% $$$ 
% $$$ Q = 0;
% $$$ 
% $$$ for i=1:n
% $$$   for j=1:n
% $$$     
% $$$    if not(isequal(mod(i),mod(j)))
% $$$      continue
% $$$    end
% $$$ 
% $$$    Q = Q + (adj(i,j) - sum(adj(i,:))*sum(adj(j,:))/(2*m))/(2*m);
% $$$    
% $$$   end
% $$$ end

% Returns the total number of edges given the adjacency matrix
% Valid for both directed and undirected, simple or general graph
% INPUTs: adjacency matrix
% OUTPUTs: m - total number of edges/links
% Other routines used: selfloops.m, issymmetric.m
% GB, Last Updated: October 1, 2009

function m = numedges(adj)

sl=selfloops(adj); % counting the number of self-loops

if issymmetric(adj) & sl==0    % undirected simple graph
    m=sum(sum(adj))/2; 
    return
elseif issymmetric(adj) & sl>0
    sl=selfloops(adj);
    m=(sum(sum(adj))-sl)/2+sl; % counting the self-loops only once
    return
elseif not(issymmetric(adj))   % directed graph (not necessarily simple)
    m=sum(sum(adj));
    return
end
end

% counts the number of self-loops in the graph
% INPUT: adjacency matrix
% OUTPUT: interger, number of self-loops
% Last Updated: GB, October 1, 2009

function sl=selfloops(adj)

sl=sum(diag(adj));
end

% Checks whether a matrix is symmetric (has to be square)
% Check whether mat=mat^T
% INPUTS: adjacency matrix
% OUTPUTS: boolean variable, {0,1}
% GB, October 1, 2009

function S = issymmetric(mat)

S = false; % default
if mat == transpose(mat); S = true; end
end
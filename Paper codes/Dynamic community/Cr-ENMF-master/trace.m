function [Zt] = trace( Z )
%TRACE writed by lanlan
% maximum matching degree
%   Z: 2-D, the label of nodes over time, 0 is unexisting
%   Zt: the modified label of nodes over time, 0 is non-existent
    [T, N] = size(Z);
    Zt = zeros(T, N);
    Zt(1,:) = Z(1,:);
    for t=2:T-1;
        Zt(t,:) = match(Zt(t-1,:), Z(t,:));
    end
    Zt(T,:) = match(Zt(T-1,:), Z(T,:));
end

function [L] = match(L1, L2)
%compute the similarity of communities using MI
L = zeros(size(L2));  % 1 * N
% [0] is none node
comms1 = setdiff(intersect(L1, L1), [0]); % find list difference of two vector
comms2 = setdiff(intersect(L2, L2), [0]); % find id of community
comms_max = max(comms1);  % new index
[~,comms1_num] = size(comms1);
[~,comms2_num] = size(comms2);
if comms1_num == 0 | comms2_num == 0;
    L = L2;
else
    M = zeros(comms1_num, comms2_num);
    for i=1:comms1_num;
        c1 = comms1(i); % get id in t-1
        if c1 == 0;
            continue;
        end;
        for j=1:comms2_num;
            c2 = comms2(j);
            if c2 == 0;
                continue;
            end;
            node1 = find(L1==c1);
            node2 = find(L2==c2);
            [~, a] = size(intersect(node1, node2));
            M(i, j) = a;  % the number of overlapped nodes
        end;
    end;
%     disp(size(M))
    sum_j = sum(M,1); % sum for column(i), [1, k] matrix
    sum_i = sum(M,2); % sum for row(j), [k, 1] matrix
    sum_ij = sum(sum(M)); % sum for all
    
    MI = zeros(size(M));
    for i=1:comms1_num
        for j=1:comms2_num
            MI(i,j) = M(i,j)/sum_i(i, 1)*log(M(i,j)*sum_ij/(sum_i(i, 1)*sum_j(1,j)));
            if isnan(MI(i,j)) == 1
                MI(i,j) = 0;
            end
        end
    end
%     disp(MI)
    [values, cols] = max(MI,[],2); % maximum matching from Ct-1 to Ct
    % cols: t-1时刻的社团演化成了t时刻的哪个社团
    new_num = 0; % count new community in Ct
    for i=1:comms2_num; % for Ct
        c2 = comms2(i);  % c2 never reappear again in Ct
        if c2 == 0;
            continue;
        end;
        % 演变到c2社团的社团是c1
        tracing = find(cols==i);  % cols is index of c2 and tracing is index of c1
        % new community
        if isempty(tracing) | values(tracing)==0
            new_num = new_num + 1;
            cx = comms_max + new_num;
            L(1, find(L2==c2))=cx;
        else  % 如果t时刻有多个社团属于t-1时刻的某个社团（分裂）
            [v, r] = max(values(tracing));  % 追踪最大的那个
            c1 = comms1(tracing(r));
%             find(L2==c2);
            L(1, find(L2==c2)) = c1; % exchange
        end;
    end;

end;
end


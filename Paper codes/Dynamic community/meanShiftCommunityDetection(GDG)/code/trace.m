function [Zt] = trace( Z )
%TRACE writed by lanlan
% maximum matching degree
%   Z: 2-D, the label of nodes over time, 0 is unexisting
%   Zt: the modified label of nodes over time, 0 is non-existent
    [T, N] = size(Z);
    Zt = zeros(T, N);
    Zt(1,:) = Z(1,:);
    for i=2:T-1;
        Zt(i,:) = match(Zt(i-1,:), Z(i,:));
    end
    Zt(T,:) = match(Zt(T-1,:), Z(T,:));
end

function [L] = match(L1, L2)
%compute the similarity of communities
L = zeros(size(L2));
comms1 = setdiff(intersect(L1, L1), [0]);
comms2 = setdiff(intersect(L2, L2), [0]);
comms_max = max(comms1);  % new index
[~,comms1_num] = size(comms1);
[~,comms2_num] = size(comms2);
if comms1_num == 0 | comms2_num == 0;
    L = L2;
else
    M = zeros(comms1_num, comms2_num);
    for i=1:comms1_num;
        c1 = comms1(i);
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
            [~, b] = size(union(node1, node2));
            M(i, j) = a/b;  % Jaccard index
        end;
    end;
    [values, cols] = max(M,[],2); % maximum matching
    new_num = 0;
    % cols: t-1时刻的社团演化成了t时刻的哪个社团
    for i=1:comms2_num;
        c2 = comms2(i);
        if c2 == 0;
            continue;
        end;
        % 演变到c2社团的社团是c1
        tracing = find(cols==i);
        % new community
        if isempty(tracing) | values(tracing)==0
            new_num = new_num + 1;
            cx = comms_max + new_num;
            L(1, find(L2==c2))=cx;  % change in 2020/5/9
        else  % 如果t时刻有多个社团属于t-1时刻的某个社团（分裂）
            [v, r] = max(values(tracing));  % 追踪最大的那个
            c1 = comms1(tracing(r));
            find(L2==c2);
            L(1, find(L2==c2)) = c1; % exchange
        end;
    end;

end;
end


%If you use this code please don't forget to cite my paper
% @article{mahmood2017using,
%   title={Using geodesic space density gradients for network community detection},
%   author={Mahmood, Arif and Small, Michael and Al-Maadeed, Somaya Ali and Rajpoot, Nasir},
%   journal={IEEE transactions on knowledge and data engineering},
%   volume={29},
%   number={4},
%   pages={921--935},
%   year={2017},
%   publisher={IEEE}
% }

function outpt = SimpleMSv1(inpt,meand)
[num,~] = size(inpt);
outpt=inpt;
for i = 1:num
    err = 1;
    while (err > 1e-3)
        cp = outpt(i,:);
        neigh=1/num* (sum((inpt-ones(num,1)*cp).^2,2));
%         mne=norm(neigh);neigh=neigh/mne;
%         mcp=norm(cp);cp=cp/mcp;
         dist =cp'.*cp'+ neigh;
 %        dist =neigh;
%         [sortD,~] = sort(dist);
%         bw = sortD(KthNeigh);
%         if bw>sigEstimate
            bw=meand;
%         end
        kd = exp(-dist ./ (bw^2));
        numr = kd' * inpt;
        den = sum(kd);
        np = numr/den;
        outpt(i,:) = np;
        err = abs(cp - np);
    end
end
end





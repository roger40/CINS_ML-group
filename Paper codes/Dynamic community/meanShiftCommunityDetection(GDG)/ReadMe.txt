This code is implementation of paper by the first author Arif Mahmood:

@article{mahmood2017using,
  title={Using geodesic space density gradients for network community detection},
  author={Mahmood, Arif and Small, Michael and Al-Maadeed, Somaya Ali and Rajpoot, Nasir},
  journal={IEEE transactions on knowledge and data engineering},
  volume={29},
  number={4},
  pages={921--935},
  year={2017},
  publisher={IEEE}
}

Please run the mainFunction.m Matlab file in the code folder.
Some networks are placed in data folder. 
Select one of the networks in mainFunction.m to see the results.
For result evaluation, you can use the provided modularity function or you may use Normalized Mutual Information (NMI) from some other source.
If you have  valid license of Bioinformatics Toolbox then please use the function 
feat=graphallshortestpaths(sparse(Adj),'DIRECTED',false);
If you do not have this licensed toolbox then use the opensource function: 
feat=allspath(Adj);
Please note that for networks with few thousands nodes, Matlab function is better implemented.

Hope you will enjoy this simple code. 
For any questions, please send me an email: rfmahmood@gmail.com

 
%Copyright (c) 2017, Sadegh Salesi and Georgina Cosma. All rights reserved.

function  mutpop=mutation(pop,nmut,popsize,nvar)

for n=1:nmut
    
i=randi([1 popsize]);  

j1=randi([1 nvar]);
% j2=randi([j1+1 nvar]);

if pop(i,j1)>=0.5
    pop(i,j1)=0;
else
    pop(i,j1)=1;
end

mutpop=pop;

end

end
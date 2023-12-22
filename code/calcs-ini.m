%%%%%%%%%%%%%%
%%% resampling 50 Hz, t, ax, ay, az --> does not work, it is more likely that the time values are not correct
%%%%%%%%%%%%%%

rax=[]; 
ray=[]; 
raz=[];
time=t-t(1);
index=1;

for i=1:length(time)-1
j=i;
while (index*20<time(j) && j>1)
j = j-1;
end

if (index*20>=time(j) && index*20<=time(j+1))
diftime=time(j+1)-time(j);
a1=(time(j+1)-index*20)/diftime;
a2=1-a1;
tmp = ax(j)*a1+ax(j+1)*a2;
rax=[rax tmp];
tmp = ay(j)*a1+ay(j+1)*a2;
ray=[ray tmp];
tmp = az(j)*a1+az(j+1)*a2;
raz=[raz tmp];
index = index+1;
end
end


av=zeros(length(ax),1);
averageval=50;
for i=averageval+1:length(ax)-averageval
gx=mean(ax(i-averageval:i+averageval));
gy=mean(ay(i-averageval:i+averageval));
gz=mean(az(i-averageval:i+averageval));
modg=sqrt(gx^2+gy^2+gz^2);
gx=gx/modg;
gy=gy/modg;
gz=gz/modg;
av(i)=ax(i)*gx+ay(i)*gy+az(i)*gz;
end
av=av(averageval:length(ax)-averageval);

av15=av;


av=[av01 av02 av03 av04 av05 av06 av07 av08 av09 av10 av11 av12 av13 av14 av15]';

save walk_all.csv av -ascii
save run_04.csv av -ascii
save run_15.csv av -ascii


figure(1)
plot(ax(500:700))
figure(2)
plot(rax(500:700))


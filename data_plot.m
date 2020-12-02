array = csvread('output.csv');
array = smoothdata(array);
plot(array)

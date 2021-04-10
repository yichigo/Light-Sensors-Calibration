
% load spatial radiance data
folder = "spatial";
node_id = "10004098";
dir_out = "../figures/" + folder + "/";
dir_data = "../data/";
fn_data = dir_data + "driving_" + node_id + ".csv";

df = readtable(fn_data, 'VariableNamingRule','preserve');
df.longitude = - df.longitude;
df.UTC.TimeZone = "UTC";

df.LocalTime = df.UTC;
df.LocalTime.TimeZone = "America/Chicago";
df.date = string(datestr(df.LocalTime,'yyyy/mm/dd'));


% merge landsat
sd = '2020-02-08';
ed = '2020-02-09';
fn = ['Landsat8_SR_DFW_',sd,'_',ed];
filename = ['../data/Landsat8/',fn,'.tif'];

[A,R] = readgeoraster(filename, 'OutputType', 'double');
proj = R.ProjectedCRS;
info = geotiffinfo(filename);

[x1,y1] = projfwd(proj, df.latitude,df.longitude); % lat,lon to x,y
[rows1,cols1] = map2pix(info.RefMatrix,x1,y1);  % x,y to row,col
rows1 = round(rows1); % to int index
cols1 = round(cols1); % to int index

for i = 1:4
    df.(['surface reflectance band ',int2str(i)]) = A(sub2ind(size(A),rows1,cols1, ones(size(rows1))*i));
end

fn_out = dir_data + "driving_" + node_id + "_landsat.csv";
writetable(df,fn_out)


% plot parameters
output_size = [1600 1200];
resolution = get(0,'ScreenPixelsPerInch');
fontsize = 112*24/resolution;



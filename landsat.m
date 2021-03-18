% close all
% clear all%

sd = '2020-02-08';
ed = '2020-02-09';
fn = ['Landsat8_SR_DFW_',sd,'_',ed];
filename = ['../data/Landsat8/',fn,'.tif'];

[A,R] = readgeoraster(filename, 'OutputType', 'double');
proj = R.ProjectedCRS;
info = geotiffinfo(filename);

height = info.Height; % Integer indicating the height of the image in pixels
width = info.Width; % Integer indicating the width of the image in pixels

[cols, rows] = meshgrid(1:width, 1:height);
[x,y] = pix2map(info.RefMatrix, rows, cols);
[lat,lon] = projinv(info,x,y);



% [x1,y1] = projfwd(proj,lat,lon);
% [rows1,cols1] = map2pix(info.RefMatrix,x1,y1)
% rows1 = round(rows1);
% cols1 = round(cols1);

fontsize = 12;
geoshow(lat,lon, A(:,:,4:-1:2)*0.0001*3); % here *3 is for brighter
% geobasemap satellite;
title({"Surface Reflectance of Bands: 4 (Red), 3 (Green), 2 (Blue)";...
       "UTC Time: " + strrep(sd,'-','/') + '-' + strrep(ed,'-','/')});
xlim([lon(1), lon(end)]);
ylim([lat(end), lat(1)]);
set(gca,'FontSize', fontsize);
print(gcf, '-dpng', "../figures/spatial/Landsat8/"+ fn+ ".png");
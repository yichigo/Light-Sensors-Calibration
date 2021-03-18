% close all
% clear all%
fontsize = 13;
vars = ["reflectivity",...
        "velocity",...
        "spectrum_width",...
        "differential_phase",...
        "differential_reflectivity",...
        "cross_correlation_ratio",...
        "ROI"];

year = '2020';
month = '02';
day = '01';
hour = '020000';

year_month = [year, '-', month];
year_month_day = [year_month, '-', day];
filename = ['/Volumes/Backup Plus/NEXRAD/halfyear/', year_month, '/', year_month_day, '/', hour, '/', year_month_day, '_', hour, '.nc'];

% merge NEXRAD
vars = ["reflectivity",...
        "velocity",...
        "spectrum_width",...
        "differential_phase",...
        "differential_reflectivity",...
        "cross_correlation_ratio",...
        "ROI"];

% design a function:  lat, lon --> x,y --> row, col
filename = ['/Volumes/Backup Plus/NEXRAD/halfyear/2020-01/2020-01-01/000000/2020-01-01_000000.nc'];
origin_latitude = ncread(filename, 'origin_latitude');
origin_longitude = ncread(filename, 'origin_longitude');
xs0 = ncread(filename, 'x');
ys0 = ncread(filename, 'y');
dx = xs0(2) - xs0(1);
dy = ys0(2) - ys0(1);
R = 6371*1000;

lats0 = rad2deg(ys0/R) + origin_latitude;
lons0 = rad2deg(xs0/R/cos(deg2rad(origin_latitude))) + origin_longitude;

[lons, lats] = meshgrid(lons0, lats0);

year = '2020';
for i_month = [1:3]
    month = num2str(i_month,'%02d');
    year_month = [year, '-', month];
    
    for i_day = [1:31]
        day = num2str(i_day,'%02d');
        year_month_day = [year_month, '-', day];
        year_month_day

        for i_hour = [0:23]
            hour = [num2str(i_hour,'%02d'),'0000'];
            year_month_day_hour = [year_month_day, ' ', num2str(i_hour,'%02d'),':00'];
            
            filename = ['/Volumes/Backup Plus/NEXRAD/halfyear/', year_month, '/', year_month_day, '/', hour, '/', year_month_day, '_', hour, '.nc'];
            
            for i_var = 1:length(vars)
                var = vars(i_var);
                values = ncread(filename, var);
%                 values = fillmissing(values,'constant',-9999.0);
                
                for i = 1:11
                    var_i = var + ' ' + int2str(i-1) + 'km';
                    var_i = char(var_i);
                    values_i = values(:,:,i).';
                    
                    
                    latlim = [lats(1) lats(end)];
                    lonlim = [lons(1) lons(end)];
                    states = geoshape(shaperead('usastatehi', 'UseGeoCoords', true));

                    figure
                    ax = usamap(latlim, lonlim);
                    fillColor = [1.0 1.0 1.0];
                    setm(ax, 'FFaceColor', fillColor)
                    geoshow(states, 'FaceColor', fillColor)
                    t = geoshow(lats,lons, values_i,...
                            'DisplayType','texturemap'); % texturemap, surface
                    t.AlphaDataMapping = 'none'; % interpet alpha values as transparency values
                    t.FaceAlpha = 'texturemap'; % Indicate that the transparency can be different each pixel
                    alpha(t,double(~isnan(values_i)));
                    colormap jet
                    colorbar
                    title({[upper(var_i(1)), var_i(2:end)];...
                           "Texas, UTC Time: " + year_month_day_hour});
                    grid on
                    set(gca,'FontSize', fontsize);
                    print(gcf, '-dpng', "../figures/spatial/Landsat8/"+ fn+ ".png");
                end
            end
        end
    end
end


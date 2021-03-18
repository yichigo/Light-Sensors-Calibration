output_size = [1600 1200];
resolution = get(0,'ScreenPixelsPerInch');
fontsize = 48*24/resolution;

folder = "AQI";
dir_data = "../data/" + folder + "/";
dir_out = "../figures/" + folder + "/";


fn_list = dir(dir_data);
df = [];
for i = 1:length(fn_list)
    fn_data = fn_list(i).name;
    if (length(fn_data) < 5)
        continue;
    end
    
    df1 = readtable(dir_data + fn_data);
    df1 = df1(:,{'Date','SITE_LATITUDE','SITE_LONGITUDE','DailyMeanPM2_5Concentration'});
    if isempty(df)
        df = df1;
    else
        df = [df;df1];
    end
end

df = df(df.Date == "01/31/2020",:);

lat = df.SITE_LATITUDE;
lon = df.SITE_LONGITUDE;
values = df.DailyMeanPM2_5Concentration;
t = df.Date;


geoscatter(lat,lon, fontsize, values,'o', 'filled');
geobasemap satellite;
colormap jet;
c = colorbar;
c.Label.String = "PM2.5" + " / " + "μg/m^3";
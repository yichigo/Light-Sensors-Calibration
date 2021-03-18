output_size = [1600 1200];
resolution = get(0,'ScreenPixelsPerInch');
fontsize = 112*24/resolution;

folder = "spatial";
node_id = "10004098";
dir_out = "../figures/" + folder + "/";
dir_data = "../data/";
fn_data = dir_data + "driving_" + node_id + ".csv";

df = readtable(fn_data);
df.longitude = - df.longitude;
df.UTC.TimeZone = "UTC";

df.LocalTime = df.UTC;
df.LocalTime.TimeZone = "America/Chicago";
df.date = string(datestr(df.LocalTime,'yyyy/mm/dd'));

features = ["Illuminance"];
for i = 360:780
    var = string(i)+ "nm";
    df.Properties.VariableNames{"x"+var} = char(var);
    features = [features, var];
end

% filter lat, long
lat_lab = 32.9922;
long_lab = -96.7579;

delta_latlong = 0.002;

iwant = (abs(df.latitude - lat_lab) + abs(df.longitude - long_lab))...
        > delta_latlong;
df = df(iwant,:);

dates = unique(df.date, 'rows');

for i = 1:length(dates)
    date = dates(i);
    disp(date);
    
    iwant = (df.date == date) & (df.LocalTime.Hour >= 7) & (df.LocalTime.Hour <= 18);
    df1 = df(iwant,:);
    if height(df1) == 0
        continue;
    end
    
    dir_day = dir_out + datestr(date, "yyyymmdd") + "/";
    if ~exist(dir_day, 'dir')
        mkdir(dir_day);
    end
    
    for i = 1:length(features)
        var = features(i);
        if (i>1) & (mod((i-2),20) ~= 0)
            continue
        end
        
        datetimes = df1.LocalTime;
        datetime_start = datetimes(1);
        datetime_end = datetimes(length(datetimes));

        time_start = datestr(datetime_start, 'HH:MM:SS');
        time_end = datestr(datetime_end, 'HH:MM:SS');

        lon = df1.longitude;
        lat = df1.latitude;

        if (var == "Illuminance")
            unit = "lux";
            title_var = var;
        elseif (var == "Zenith")
            unit = "^\circ";
            title_var = var;
        else % wavelengths like "450nm"
            unit = "W/m^2";
            title_var = "Irradiance at " + var;
        end

        values = df1.(var);

        % Visualize Spatial Irradiance
        geoscatter(lat,lon, fontsize, values,'o', 'filled');
        geobasemap satellite;
        title({"Spatial Variation of " + title_var;...
                "Local Time: " + date + "  " + ...
                time_start + "-" + time_end});
        colormap jet;
        c = colorbar;
        c.Label.String = var + " / " + unit;
        caxis([0, inf]);
        set(gca,'FontSize', fontsize);
        set(gcf,'paperunits','inches','paperposition',[0 0 output_size/resolution]);
        set(gcf,'Visible', 'off');
        print(gcf, '-dpng', ['-r' num2str(resolution)], ...
            dir_day +"/spatial_"+ var + "_"...
            + datestr(datetime_end, 'yyyy_mm_dd') + ".png");

        % Visualize Termoral Irradiance
        scatter(datetimes, values, fontsize, values, 'o', 'filled');
        title({ title_var;...
                "Local Time: " + date});
        grid on
        colormap jet;
        c = colorbar;
        c.Label.String = var + " / " + unit;
        caxis([0, inf]);
        ylim([0 inf])
        set(gca,'FontSize', fontsize);
        set(gcf,'paperunits','inches','paperposition',[0 0 output_size/resolution]);
        set(gcf,'Visible', 'off');
        print(gcf, '-dpng', ['-r' num2str(resolution)], ...
            dir_day +"/localtime_"+ var + "_"...
            + datestr(datetime_end, 'yyyy_mm_dd') + ".png");
        
    end
end

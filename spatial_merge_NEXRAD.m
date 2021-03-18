
% load spatial radiance data
folder = "spatial";
node_id = "10004098";
dir_out = "../figures/" + folder + "/";
dir_data = "../data/";
fn_data = dir_data + "driving_" + node_id + "_landsat.csv";

df = readtable(fn_data, 'VariableNamingRule','preserve');

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


year = '2020';
for i_month = [1:3]
    month = num2str(i_month,'%02d');
    year_month = [year, '-', month];
    
    iwant_month = (df.UTC.Month == i_month);
    if sum(iwant_month) == 0
        continue;
    end
    
    for i_day = [1:31]
        day = num2str(i_day,'%02d');
        year_month_day = [year_month, '-', day];
        
        iwant_day = (iwant_month & (df.UTC.Day == i_day));
        if sum(iwant_day) == 0
            continue;
        end
        
        year_month_day

        for i_hour = [0:23]
            hour = [num2str(i_hour,'%02d'),'0000'];
            
            iwant_hour = (iwant_day & (df.UTC.Hour == i_hour));
            if sum(iwant_hour) == 0
                continue;
            end
            
            filename = ['/Volumes/Backup Plus/NEXRAD/halfyear/', year_month, '/', year_month_day, '/', hour, '/', year_month_day, '_', hour, '.nc'];
            
            lats = df.latitude(iwant_hour);
            lons = df.longitude(iwant_hour);
            
            xs = R*deg2rad(lons - origin_longitude) * cos(deg2rad(origin_latitude));
            ys = R*deg2rad(lats - origin_latitude);
            rows = (ys - ys0(1)) / dy + 1;
            cols = (xs - xs0(1)) / dx + 1;
            
            rows_floor = floor((ys - ys0(1)) / dy) + 1;
            cols_floor = floor((xs - xs0(1)) / dx) + 1;
            
            rows_ceil = ceil((ys - ys0(1)) / dy) + 1;
            cols_ceil = ceil((xs - xs0(1)) / dx) + 1;
            
            for i_var = 1:length(vars)
                var = vars(i_var);
                values = ncread(filename, var);
                if var ~= "ROI"
                    values = fillmissing(values,'constant',-9999.0);
                else
                    values = fillmissing(values,'constant',1e+20);
                end
                
                for i = 1:11
                    var_i = var + ' ' + int2str(i-1) + 'km';
                    
                    % Linear nterpolation
                    values_ff = values(sub2ind(size(values), rows_floor, cols_floor, ones(size(rows))*i));
                    values_fc = values(sub2ind(size(values), rows_floor, cols_ceil,  ones(size(rows))*i));
                    values_cf = values(sub2ind(size(values), rows_ceil,  cols_floor, ones(size(rows))*i));
                    values_cc = values(sub2ind(size(values), rows_ceil,  cols_ceil,  ones(size(rows))*i));
                    
                    values_f = (values_ff.*(cols-cols_ceil) + values_fc.*(cols-cols_floor))./(cols_ceil - cols_floor);
                    values_c = (values_cf.*(cols-cols_ceil) + values_cc.*(cols-cols_floor))./(cols_ceil - cols_floor);
                    
                    df.(var_i)(iwant_hour) = (values_f.*(rows-rows_ceil) + values_c.*(rows-rows_floor))./(rows_ceil - rows_floor);
                    
                end
            end
        end
    end
end

fn_out = dir_data + "driving_" + node_id + "_landsat_NEXRAD.csv";
writetable(df,fn_out)



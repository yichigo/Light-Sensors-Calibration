import os

node_list = ['001e063059c2', '001e06305a61', '001e06305a6c', '001e06318cd1', '001e06323a05', '001e06305a57', '001e06305a6b', '001e06318c28', '001e063239e3', '001e06323a12']

for node_id in node_list:
    print('##############')
    print(node_id)
    os.system('python data_lightsensors.py '+ node_id )


# Report Project Mines ML&DL 2020
Author: Vladimir Chizhevskiy

## Data preparation 
There were not so much changes applied to the data
- My pipeline for data retreiving was to just take mean traffic for the area and then to create sequences of five hours after the current value so we could use the obtained data for RNN
- I used not the volume of traffic but its difference from the hour to hour starting the first value
```
hourly_locations[i].diff().fillna(hourly_locations[i][0]).astype(np.int64)
```
- I applied this pipeline for two experiments:
1) predicted the traffic for the whole city of Austin
2) predicted the traffic for each of the given locations in Austin 



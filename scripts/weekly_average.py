import csv
import datetime

weekly_averages = [[]]
current_week = datetime.date(2015,2,18)
week_values = [current_week]
week = datetime.timedelta(days=7)
i = 0

## Replace csv filename for different companies
with open('fund/data/google/sentiment/google_sentiment.csv','rt') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        today = datetime.date(int(row['year']),int(row['month']),int(row['day']))
        if today>=current_week+week:
            current_week = today
            week_values.append(today)
            weekly_averages.append([])
            i+=1
        weekly_averages[i].append(row['v2tone'])

weekly_averages_calculated = []

total = float(0)
total_count = float(0)

for week in weekly_averages:
    sum = float(0)
    count = float(len(week))
    for value in week:
        sum+=float(value)
        total+= float(value)
        total_count+=float(1)
    weekly_averages_calculated.append(sum/count)
    
print('global average: '+str(total/total_count))

## Replace csv filename for different companies
with open('fund/data/google/sentiment/google_averages.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['Week','Average'])
    for i in range(0,len(weekly_averages_calculated)):   
        writer.writerow([week_values[i], weekly_averages_calculated[i]])


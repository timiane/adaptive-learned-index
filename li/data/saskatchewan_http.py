from datetime import datetime
import re


origin_date = "01/Jun/1995:00:00:00 -0600"
origin_date_time = datetime.strptime(origin_date, "%d/%b/%Y:%H:%M:%S %z")
data = []

f = open("UofS_access_log", "r", encoding="latin1")

# all_lines = f.readlines()

for line in f:
    try:
        date = re.search('\[.*\s.{5}\]', line)
        data_string = date.group(0).strip("*[*]")
        date_time = datetime.strptime(data_string, "%d/%b/%Y:%H:%M:%S %z")
        data.append((date_time - origin_date_time).total_seconds())
    except ValueError:
        print(line)
        print(data_string)
f.close()


with open("saskatachewan_data.csv", "w+") as file_writer:
    for element in data:
        file_writer.write(str(element) + "\n")
        file_writer.close()





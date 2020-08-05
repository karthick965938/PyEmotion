import csv
from pathlib import Path
# with open('file.csv', mode='w') as csv_file:
#     fieldnames = ['emp_name', 'dept', 'birth_month']
#     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

#     writer.writeheader()
#     writer.writerow({'emp_name': 'John Smith', 'dept': 'Accounting', 'birth_month': 'November'})
#     writer.writerow({'emp_name': 'Erica Meyers', 'dept': 'IT', 'birth_month': 'March'})
# header = ['Emotion', 'Time']
# some_list = ['Sad,2020-07-18 15:30:26.406681', 'Sad,2020-07-18 15:30:26.697549', 'Sad,2020-07-18 15:30:26.884139', 'Sad,2020-07-18 15:30:27.026544', 'Sad,2020-07-18 15:30:27.165891', 'Sad,2020-07-18 15:30:27.314882', 'Sad,2020-07-18 15:30:27.451950', 'Sad,2020-07-18 15:30:27.606951']
# my_file = Path("test.csv")
# if my_file.is_file():
# 	with open('test.csv', 'a', newline ='') as file:
# 	    writer = csv.writer(file, delimiter=',')
# 	    for j in some_list:
# 	        writer.writerow(j.split(','))
# else:
# 	with open('test.csv', 'wt', newline ='') as file:
# 	    writer = csv.writer(file, delimiter=',')
# 	    writer.writerow(i for i in header)
# 	    for j in some_list:
# 	        writer.writerow(j.split(','))
file  = open('emotion_details.csv', "r")
a = {'Angry' : 0, 'Disgust' : 0, 'Fear' : 0, 'Happy' : 0, 'Sad' : 0, 'Surprise' : 0, 'Neutral' : 0, 'NoFace' : 0, 'Discussion' : 0 }
read = csv.reader(file)
for i, row in enumerate(read):
  if row[0] in a:
  	a[row[0]] += 1
print(a)
import csv

#Purpose of this file is to modify the current training data, as it needs an overhaul to include the framerate


postfix = ".csv"
prefix = "Train"
for i in range(0,400):
    print(i)
    newRowsList = list()
    with open((prefix+str(i)+postfix),'rt',encoding = 'utf-8') as f:
        reader = csv.reader(f)#Open the csv
        for row in reader:
            newRow = row[:] + [0 for _ in range(22197 - len(row))]
            newRowsList.append(newRow)


    with open((prefix+str(i)+postfix),'w',newline ='') as f:
        write = csv.writer(f,delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)

        for r in newRowsList:
            write.writerow(r)
        
             



                                                
import csv
import os

def readCSV():
    first = True
    with open("sql_query_data.csv",newline="") as csvfile:
        csvreader = csv.reader(csvfile,dialect=csv.excel_tab)
        for line in csvreader:
            row = line[0].split(",")
            if first:
                first = False
                continue
            if not row[0]: continue
            row[0]=row[0].lower()
            row[1]=row[1].lower()
            row[5]=formatTime(row[5])
            row[6]=formatTime(row[6])
            if row[5] == "skip": continue
            writeSQL(row)

def writeSQL(row):
    '''
    0:org_abbrev
    1:earnings_period
    2:org_keyword1
    3:org_keyword2
    4:doc_id
    5:start_date
    6:end_date
    '''
    file_name = row[0]+"_"+row[1]+".sql"
    if row[4]: dir_name = row[4]
    else: dir_name = row[0].lower()
    dest_dir = os.path.join(os.getcwd(),dir_name,"sql")
    print(dest_dir)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    with open(os.path.join(dest_dir,file_name),"w") as f:
        print("writing"+row[0]+"_"+row[1]+".sql")
        f.write("SELECT date, sourcecommonname, documentidentifier, "
                "v2tone, organizations, v2organizations FROM [gdelt-bq:gdeltv2.gkg] ")
        f.write("WHERE (organizations like '%"+row[2]+"%'")
        if row[3]:
            f.write(" OR v2organizations like '%"+row[3]+"%'")
        f.write(") AND date> "+row[5]+"AND date<"+row[6]+" ")
        if row[4]:
           f.write("AND documentidentifier like '%"+row[4]+"%' ")
        f.write("AND (themes like '%EPU_ECONOMY_HISTORIC%' OR themes like '%ECON_EARNINGSREPORT%');")

def formatTime(str):
    d = str.split("/")
    if int(d[2]) < 15: return "skip"
    elif int(d[2]) == 15:
        if int(d[0]) < 2: return "skip"
    ret = "20"+d[2]
    if len(d[0]) == 1: ret += "0"+d[0]
    else: ret += d[0]
    if len(d[1]) == 1: ret += "0"+d[1]
    else: ret += d[1]
    return ret+"000000"

if __name__ == "__main__":
    readCSV()

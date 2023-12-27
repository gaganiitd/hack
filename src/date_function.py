from datetime import datetime, timedelta
def dates(d):
    dates = []
    start = datetime.now()
    dur = timedelta(days=1)
    dates.append(str(start.year) + '-' + str(start.month) + '-' + str(start.day))
    
    for i in range(0, d):
        start = start + dur
        dates.append(str(start.year) + '-' + str(start.month) + '-' + str(start.day))
    
    return dates



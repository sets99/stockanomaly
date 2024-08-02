#for function testing purposes

from datetime import date, datetime, timedelta

date1 = date.today()
date2 = date.today() - timedelta(days=4)
print((date2 - date1).days)
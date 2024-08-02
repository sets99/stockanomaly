from datetime import date, timedelta
import math

stockperiodend = date.today()
stockperiodstart = stockperiodend - timedelta(days=365)


def getMeanReturn(date1, date2):
    days = (date2 - date1).days
    sum = 0
    for i in range(2, days):
        date = date1 + timedelta(days=i)
        sum = sum + math.log(closingprice(date)/closingprice(date - timedelta(days=1)))

    return sum/days


def getVariance(date1, date2):
    days = (date2 - date1).days
    sum = 0
    mean = getMeanReturn(date1, date2)
    
    for i in range(2, days):
        date = date1 + timedelta(days=i)
        sum = sum + (mean - math.log(closingprice(date)/closingprice(date - timedelta(days=1))))**2

    variance = sum/(days-1)
    volatility = math.sqrt(variance)
    return volatility


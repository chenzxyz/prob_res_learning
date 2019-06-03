import datetime

# ts = datetime.datetime.now()
#
# tid = '{}_{}_{}_{}_{}'.format(ts.year, ts.month, ts.day, ts.hour, ts.minute)

t = datetime.datetime.now().strftime('%m%d_%H%M')

print(t)
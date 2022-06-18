# test
from psutil import disk_usage
disk='/media/arash/7bee828d-5758-452e-956d-aca000de2c81'
hdd=disk_usage(disk)
total,used,free=hdd.total / (2**30),hdd.used / (2**30),hdd.free / (2**30)
print ("Total: %d GiB" % total)
print ("Used: %d GiB" % used)
print ("Free: %d GiB" % free)
if free<10:
    raise NameError('disk is almost full')

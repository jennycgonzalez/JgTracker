time ./jgtracker track /media/disk2/jgthesis-videos/Lemming/ -1

** (Target:10210): WARNING **: Couldn't connect to accessibility bus: Failed to connect to socket /tmp/dbus-Rw8TeFRaCV: Connection refused
PROFILE: interrupts/evictions/bytes = 2071/652/143576

real	0m19.549s
user	0m23.465s
sys	0m0.948s

google-pprof --web jgtracker /tmp/jgthesis.prof

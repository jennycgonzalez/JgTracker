time ./jgtracker track /media/disk2/jgthesis-videos/Lemming/ -1

** (Target:11110): WARNING **: Couldn't connect to accessibility bus: Failed to connect to socket /tmp/dbus-Rw8TeFRaCV: Connection refused
PROFILE: interrupts/evictions/bytes = 60365/43762/3424888

real	5m11.443s
user	17m27.459s
sys	0m7.133s

google-pprof --web jgtracker /tmp/jgthesis.prof

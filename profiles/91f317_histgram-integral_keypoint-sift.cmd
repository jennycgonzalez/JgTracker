time ./jgtracker track /media/disk2/jgthesis-videos/Lemming/ -1

** (Target:11971): WARNING **: Couldn't connect to accessibility bus: Failed to connect to socket /tmp/dbus-Rw8TeFRaCV: Connection refused
PROFILE: interrupts/evictions/bytes = 7581/3195/502416

real	1m12.958s
user	1m22.667s
sys	0m0.929s

google-pprof --web jgtracker /tmp/jgthesis.prof

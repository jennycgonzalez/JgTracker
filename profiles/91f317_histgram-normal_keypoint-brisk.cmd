time ./jgtracker track /media/disk2/jgthesis-videos/Lemming/ -1

** (Target:9126): WARNING **: Couldn't connect to accessibility bus: Failed to connect to socket /tmp/dbus-Rw8TeFRaCV: Connection refused
PROFILE: interrupts/evictions/bytes = 56616/41540/3167184

real	4m32.798s
user	16m50.533s
sys	0m7.275s

google-pprof --web jgtracker /tmp/jgthesis.prof

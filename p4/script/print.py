import time

pipe = bfrt.mrac.pipe
counter = pipe.SwitchIngress.counter
counter.operation_counter_sync(callback=None)
time.sleep(3)

entries = counter.get(regex=True, print_ents=False, return_ents=True)
entries = [e.data[b'$COUNTER_SPEC_PKTS'] for e in entries]
for i, e in enumerate(entries):
    if e > 0:
        print("{}: {}".format(i, e))


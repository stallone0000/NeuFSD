CC ?= gcc
CFLAGS = -O2

MT_CFLAGS = $(CFLAGS) -pthread

DPDK_CFLAGS := $(CFLAGS) $(shell pkg-config --cflags libdpdk)
DPDK_LDFLAGS := $(shell pkg-config --libs libdpdk)

TARGETS = convert sender sender_dpdk receiver_throughput_realtime \
	  receiver_throughput_total receiver_latency

all: $(TARGETS)

convert: convert.c
	$(CC) $(CFLAGS) -o $@ $<

sender: sender.c
	$(CC) $(MT_CFLAGS) -o $@ $<

sender_dpdk: sender_dpdk.c
	$(CC) $(DPDK_CFLAGS) -o $@ $< $(DPDK_LDFLAGS)

receiver_throughput_realtime: receiver_throughput_realtime.c
	$(CC) $(DPDK_CFLAGS) -o $@ $< $(DPDK_LDFLAGS)

receiver_throughput_total: receiver_throughput_total.c
	$(CC) $(DPDK_CFLAGS) -o $@ $< $(DPDK_LDFLAGS)

receiver_latency: receiver_latency.c
	$(CC) $(DPDK_CFLAGS) -o $@ $< $(DPDK_LDFLAGS)

clean:
	rm -f $(TARGETS)

.PHONY: all clean


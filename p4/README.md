# NeuFSD-p4

P4 implementation of NeuFSD on Tofino.

## Repository Structure

- `p4/`: P4 implementation of NeuFSD and baseline methods (sample-based, MRAC, Elastic Sketch, DaVinci Sketch)
  - `common/`: Boilerplate code
  - `<method>.p4`:  Method implementation
  - `<method>_latency.p4`: Method implementation with modification for latency measurement
  - `davinci.p4`: DaVinci Sketch P4 source with a hot-flow part, three Tower layers, and three Fermat arrays.
- `profiling/`: Packet sender, receiver and related utils for measuring performance
  - `sender.c`: Multi-thread-based packet sender, taking `.pcap` file as input
  - `sender_dpdk.c`: DPDK-based packet sender, taking 5-tuple-based dataset as input
  - `receiver_throughput_realtime.c`: DPDK-based packet receiver, showing real-time throughput
  - `receiver_throughput_total.c`: DPDK-based packet receiver, showing overall throughput after running
  - `receiver_latency.c`: DPDK-based packet receiver for latency measurement
  - `convert.c`: Convert 5-tuple-based dataset to valid `.pcap` files
- `script/`: Scripts for setup and data collection
  - `setup.py`: Set up Tofino switch after starting P4 program
  - `print.py`: Check data plane counter of MRAC
  - ` collect_resource_usage.sh`: Collect resource usage logs from build directories

## Environment

- Switch side: Intel Tofino SDE 9.6.0
  - Compilation requires `bf-p4c`/`p4c-tofino`.

- Server side: DPDK 21.11.9

## How to Run

1. Set up environment on both switch and server
2. Compile and start a P4 program
3. Compile and start packet receiver and sender respectively

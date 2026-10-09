def enable_ports(ports):
    PORT_SPEED = "BF_SPEED_100G"
    FEC_TYPE = "BF_FEC_TYP_RS"
    # For more device port mapping, check bfshell->pm->`show -a`
    NAME_TO_DEV_PORT = {
	"1/0": 132,
	"3/0": 148,
	"5/0": 164,
	"7/0": 180,
    }
    
    port_table = bfrt.port.port
    for p in ports:
        try:
            port_table.add(
                DEV_PORT=NAME_TO_DEV_PORT[p],
                SPEED=PORT_SPEED,
                FEC=FEC_TYPE,
                PORT_ENABLE=True,
            )
            print("Port {} configured and enabled".format(p))
        except Exception as e:
            print("Error configuring port {}: {}".format(p, e))


print("Configuring ports...")
PORTS = ["1/0", "3/0"]
enable_ports(PORTS)

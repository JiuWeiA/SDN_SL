from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.topo import Topo
from mininet.cli import CLI
from mininet.link import TCLink


class MyTopo(Topo):
    def build(self):
        h1 = self.addHost('h1', ip='10.0.0.1/24')
        h2 = self.addHost('h2', ip='10.0.0.2/24')

        s1 = self.addSwitch('s1')
        s2 = self.addSwitch('s2')
        s3 = self.addSwitch('s3')

        self.addLink(h1, s1)
        self.addLink(s1, s2, delay='20ms')
        self.addLink(s1, s3, delay='1ms')
        self.addLink(s2, h2)
        self.addLink(s3, h2)


def run():
    topo = MyTopo()
    net = Mininet(
        topo=topo,
        controller=None,
        switch=OVSSwitch,
        link=TCLink,
        autoSetMacs=True
    )

    net.addController(
        'c0',
        controller=RemoteController,
        ip='127.0.0.1',
        port=6633
    )

    try:
        net.start()
        print("Mininet started. Type commands in CLI.")
        CLI(net)
    finally:
        net.stop()


if __name__ == '__main__':
    run()

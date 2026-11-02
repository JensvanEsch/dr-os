from joint_bus_contract import JointBusContract

from dros.servo import SimBus


class TestSimBusFollowsTheContract(JointBusContract):
    def make_bus(self, geom, start_angles, **limits):
        return SimBus(geom, start_angles=start_angles, **limits)

    def settle(self, bus):
        bus.advance(10.0)

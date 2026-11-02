from joint_bus_contract import JointBusContract

from dros.servo import MockBus


class TestMockBusFollowsTheContract(JointBusContract):
    def make_bus(self, geom, start_angles, **limits):
        return MockBus(geom, start_angles=start_angles, **limits)

from vehicle import *

print("Start:", get_vehicle_status())
print()

print("Set 22 C:", set_temperature(22))
print("Set 50 C:", set_temperature(50))
print("Seat heating 2:", set_seat_heating(2))
print("Seat heating 7:", set_seat_heating(7))

car["speed_kmh"] = 130
print("Open windows at 130 km/h:", set_windows(True))

print()
print("End:", get_vehicle_status())
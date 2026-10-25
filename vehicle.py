# ---------- The car's current state ----------
car = {
    "heater_on": False,
    "target_temp_c": 20,
    "cabin_temp_c": 14,
    "seat_heating_level": 0,
    "windows_open": False,
    "speed_kmh": 0,
    "battery_percent": 68,
    "range_km": 290,
}


# ---------- Actions the car can do ----------
def get_vehicle_status():
    return car


def set_temperature(temp_c):
    if temp_c < 16 or temp_c > 30:
        return {"ok": False, "error": "Temperature must be between 16 and 30 C"}
    car["target_temp_c"] = temp_c
    car["heater_on"] = True
    return {"ok": True, "target_temp_c": temp_c}


def set_seat_heating(level):
    if level not in [0, 1, 2, 3]:
        return {"ok": False, "error": "Seat heating level must be 0, 1, 2 or 3"}
    car["seat_heating_level"] = level
    return {"ok": True, "seat_heating_level": level}


def set_windows(open_windows):
    if open_windows and car["speed_kmh"] > 100:
        return {"ok": False, "error": "Windows cannot open above 100 km/h"}
    car["windows_open"] = open_windows
    return {"ok": True, "windows_open": open_windows}
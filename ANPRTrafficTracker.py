import os
from concurrent.futures import ThreadPoolExecutor
import anpr_multi_camera as ANPR
import gps_server as GPSServer


def startANPRCameras():
    ANPR.main()
def startGPSServer():
    os.environ["GPS_SERVER_HOST"] = "0.0.0.0"
    os.environ["GPS_SERVER_PORT"] = "5000"
    os.environ["GPS_PUBLIC_BASE_URL"] = "http://10.200.195.29:5000"
    GPSServer.main()
def startTrajectoryANPRBridge():
    pass
def startTrajectoryEngine():
    pass
def boolInput(prompt):
    truthy = {"true", "yes", "y", "1"}
    falsey = {"false", "no", "n", "0"}
    while True:
        choice = input(prompt).strip().lower()
        if choice in truthy:
            return True
        if choice in falsey:
            return False
        print("Please enter a valid choice.")

isThisGpsServer = boolInput("Run the GPS server? (y/n): ")

with ThreadPoolExecutor() as executor:
    if isThisGpsServer:
        executor.submit(startGPSServer)
    os.environ["GPS_SERVER_HOST"] = "0.0.0.0"
    os.environ["GPS_SERVER_PORT"] = "5000"
    GPS_SERVER_IP = input("Enter the GPS server IP address: ")
    os.environ["GPS_PUBLIC_BASE_URL"] = f"http://{GPS_SERVER_IP}:5000"
    executor.submit(startANPRCameras)

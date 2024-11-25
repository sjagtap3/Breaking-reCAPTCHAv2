import njord

print("Location of njord:")
print(njord.__file__)
client = njord.Client(user="sjagtap3@illinois.edu", password="Shrushti@04") #enter your username and password here
# Explicit
def connect():
    client.connect()

def disconnect():
    client.disconnect()
    print("vpn disconnected")
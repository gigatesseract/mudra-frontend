import time
import serial
 
ser = serial.Serial('/dev/ttyACM0', 9600)
 
while True:
 value = ser.readline()
 print(ser.read(1))
 print(ser.read(2))
 print("sensor reading:", value)
 time.sleep(0.5)
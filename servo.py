import serial
import sys

MODE_256 = 0
MODE_QUARTER = 1


def clamp( val, mmin, mmax ):
    if val > mmax:
        val = mmax
    elif val < mmin:
        val = mmin
    return val

class Servo:
    ser = None

    def __init__( self, com='COM6', baud = 9600, mode=MODE_256 ):
        self.ser = serial.Serial(com)
        self.ser.baudrate = baud
        self.mode = mode
        self.coords = [0,0]

    def stop( self ):
        self.ser.close()

    def shift( self, n, angle ):
        angle = clamp( self.coords[n] + angle, 0, 90)
        self.setpos( n, angle )

    def setpos( self, n, angle ):
        angle = clamp(angle, 0, 90)
        self.coords[n] = angle

        if self.mode == MODE_256:
            pos = int( 254 * angle/90.0 )
            bud=chr(0xFF)+chr(n)+chr(pos)

        elif mode == MODE_QUARTER:
            pos = int( 1000 * angle/90.0 ) + 1000
            lo = pos & 0x7F
            hi = ( pos >> 7 ) & 0x7F
            bud=chr(0x84)+chr(n)+chr(lo)+chr(hi)

        self.ser.write(bud)
        self.ser.flush()

from enum import Enum
import re
import time
import math
import copy
import struct
import snappy
from io import BytesIO
from typing import NewType
import pygame
import numpy as np
import cv2
import asyncio
from urllib.request import urlopen
import io


class FileReader(object):
    """
    Some utilities to make it a bit easier to read values out of the .dem file
    """
    def __init__(self, stream):
        self.stream = stream
        stream.seek(0, 2)
        self.size = stream.tell()
        self.remaining = self.size
        stream.seek(0)

        self.pos = 0
        self.bit_count = 0
        self.bit_val = 0

    def more(self):
        return self.remaining > 0

    def nibble(self, length):
        self.remaining -= length
        if self.remaining < 0:
            raise ValueError("Not enough data")

    def read_byte(self):
        self.nibble(1)

        return ord(self.stream.read(1))

    def read(self, length=None):
        if length is None:
            length = self.remaining

        self.nibble(length)

        return self.stream.read(length)

    def read_int32(self):
        self.nibble(4)

        return struct.unpack("i", self.stream.read(4))[0]

    def read_uint32(self):
        self.nibble(4)

        return struct.unpack("I", self.stream.read(4))[0]

    def read_boolean(self):
        return self.read_bits(1) == 1

    def next_byte(self):
        self.pos += 1
        if self.pos > self.size:
            print('nextByte: insufficient buffer ({} of {})'.format(self.pos, self.size))
        
        value = self.stream.read(1)
        value = ord(value)

        return value

    def read_byte_test(self):
        if self.bit_count == 0:
            return self.next_byte()

        return self.read_bits(8)

    def read_bytes(self, n):
        buf = bytearray()
        for i in range(n):
            data = self.read_bits(8)
            buf.extend(bytes([data]))

        return bytes(buf)

    def read_bits(self, n):
        while n > self.bit_count:
            nextByte = self.next_byte()
            self.bit_val |= nextByte << self.bit_count
            self.bit_count += 8

        x = (self.bit_val & ((1 << n) - 1))
        self.bit_val >>= n
        self.bit_count -= n
        
        return x

    def read_ubit_var_fp(self):
        if self.read_boolean():
            return self.read_bits(2)

        if self.read_boolean():
            return self.read_bits(4)

        if self.read_boolean():
            return self.read_bits(10)

        if self.read_boolean():
            return self.read_bits(17)

        return self.read_bits(31)

    def read_ubit_var_field_path(self):
        return int(self.read_ubit_var_fp())

    def read_string(self):
        buf = bytearray()
        while True:
            b = bytes([self.read_byte_test()])
            if b == b'\x00':
                break

            buf.extend(b)

        return bytes(buf)

    def read_vint32(self):
        """
        This seems to be a variable length integer ala utf-8 style
        """
        result = 0
        count = 0
        while True:
            if count > 4:
                raise ValueError("Corrupt VarInt32")

            b = self.read_byte()
            result = result | (b & 0x7F) << (7 * count)
            count += 1

            if not b & 0x80:
                return result

    def read_var_uint32(self):
        x = np.uint32(0)
        s = np.uint32(0)
        while True:
            b = np.uint32(self.read_byte_test())

            x |= (b & 0x7F) << s
            s += 7

            if ((b & 0x80) == 0) or (s == 35):
                break

        return x

    def read_var_uint64(self):
        x = 0
        s = 0
        i = 0
        while True:
            b = self.read_byte_test()
            if b < 0x80:
                if i > 9 or i == 9 and b > 1:
                    break

                return x | (int(b) << s)

            x |= int(b & 0x7f) << s
            s += 7

            i += 1

    def read_var_int32(self):
        ux = self.read_var_uint32()
        x = np.int32(ux >> 1)
        if ux & 1 != 0:
            x = ~x

        return x

    def read_ubit_var(self):
        index = self.read_bits(6)

        flag = index & 0x30
        if flag == 16:
            index = (index & 15) | (self.read_bits(4) << 4)
        elif flag == 32:
            index = (index & 15) | (self.read_bits(8) << 4)
        elif flag == 48:
            index = (index & 15) | (self.read_bits(28) << 4)

        return index

    def read_coord(self):
        value = 0.0

        intval = self.read_bits(1)
        fractval = self.read_bits(1)
        signbit = False

        if intval != 0 or fractval != 0:
            signbit = self.read_boolean()

            if intval != 0:
                intval = self.read_bits(14) + 1

            if fractval != 0:
                fractval = self.read_bits(5)

            value = float(intval) + float(fractval) * (1.0 / (1 << 5))

            if signbit:
                value = -value

        return value

    def rem_bytes(self):
        return self.size - self.pos

    def read_bits_as_bytes(self, n):
        tmp = bytearray()
        while n >= 8:
            read_value = self.read_byte_test()
            tmp.extend(bytes([read_value]))
            n -= 8

        if n > 0:
            read_value = self.read_bits(n)
            #read_value_byte = bytes(read_value)
            tmp.extend(bytes([read_value]))

        return bytes(tmp)

    def read_normal(self):
        is_neg = r.read_boolean()
        len = r.read_bits(11)
        ret = float(len) * float(1.0 / (float(1 << 11) - 1.0))

        if is_neg:
            return -ret
        else:
            return ret

    def read_3bit_normal(self):
        ret = [0.0, 0.0, 0.0]

        hasX = self.read_boolean()
        hasY = self.read_boolean()

        if hasX:
            ret[0] = self.read_normal()

        if hasY:
            ret[1] = self.read_normal()

        negZ = self.read_boolean()
        prodsum = ret[0]*ret[0] + ret[1]*ret[1]

        if prodsum < 1.0:
            ret[2] = float(math.sqrt(float(1.0 - prodsum)))
        else:
            ret[2] = 0.0

        if negZ:
            ret[2] = -ret[2]

        return ret

    def read_le_uint64(self):
        result = self.read_bytes(8)
        result = int.from_bytes(result, byteorder='little')
        #print("result: ", result)

        return result

    def read_angle(self, n):
        result = self.read_bits(n)
        result = float(result) * 360 / float(int(1 << n))

        return result

    def read_message(self, message_type, compressed=False, read_size=True):
        """
        Read a protobuf message
        """
        if read_size:
            size = self.read_vint32()
            b = self.read(size)
        else:
            b = self.read()

        if compressed:
            b = snappy.decompress(b)

        m = message_type()
        m.ParseFromString(b)

        return m, b


class FieldType:
    def __init__(self, name):
        itemCounts = {"MAX_ITEM_STOCKS":             8,
                      "MAX_ABILITY_DRAFT_ABILITIES": 48
                      }

        p = re.compile('([^\<\[\*]+)(\<\s(.*)\s\>)?(\*)?(\[(.*)\])?')
        searches = p.search(name)

        ss = searches.groups()

        self.base_type = ss[0]
        self.pointer = ss[3] == "*"
        self.generic_type = None
        self.count = 0

        if ss[2] != None:
            self.generic_type = FieldType(name=ss[2])

        if ss[5] in itemCounts:
            self.count = itemCounts[ss[5]]
        elif ss[5] != None:
            if int(ss[5]) > 0:
                self.count = int(ss[5])
            else:
                self.count = 1024


pointerTypes = {
    "PhysicsRagdollPose_t":       True,
    "CBodyComponent":             True,
    "CEntityIdentity":            True,
    "CPhysicsComponent":          True,
    "CRenderComponent":           True,
    "CDOTAGamerules":             True,
    "CDOTAGameManager":           True,
    "CDOTASpectatorGraphManager": True,
    "CPlayerLocalData":           True,
    "CPlayer_CameraServices":     True,
}


fieldTypeDecoders = {
    "bool":    "booleanDecoder",
    "char":    "stringDecoder",
    "color32": "unsignedDecoder",
    "int16":   "signedDecoder",
    "int32":   "signedDecoder",
    "int64":   "signedDecoder",
    "int8":    "signedDecoder",
    "uint16":  "unsignedDecoder",
    "uint32":  "unsignedDecoder",
    "uint8":   "unsignedDecoder",

    "CBodyComponent":       "componentDecoder",
    "CGameSceneNodeHandle": "unsignedDecoder",
    "Color":                "unsignedDecoder",
    "CPhysicsComponent":    "componentDecoder",
    "CRenderComponent":     "componentDecoder",
    "CUtlString":           "stringDecoder",
    "CUtlStringToken":      "unsignedDecoder",
    "CUtlSymbolLarge":      "stringDecoder",
}


class fieldModelEnum(Enum):
    fieldModelSimple = 0
    fieldModelFixedArray = 1
    fieldModelFixedTable = 2
    fieldModelVariableArray = 3
    fieldModelVariableTable = 4


class Serializer:
    def __init__(self, msg, s):
        self.name = msg.symbols[s.serializer_name_sym]
        self.version = s.serializer_version
        self.fields = []


class Field:
    def __init__(self, ser, f):
        self.parent_name = None

        if f.var_name_sym != 0:
            self.var_name = ser.symbols[f.var_name_sym]
        else:
            self.var_name = None
                                                                          
        if f.var_type_sym != 0:
            self.var_type = ser.symbols[f.var_type_sym]
        else:
            self.var_type = None

        if f.send_node_sym != 0:
            self.send_node = ser.symbols[f.send_node_sym]
        else:
            self.send_node = None

        if f.field_serializer_name_sym != 0:
            self.serializer_name = ser.symbols[f.field_serializer_name_sym]
        elif f.field_serializer_name_sym == 0 and self.var_type == "CBodyComponent":
            self.serializer_name = ser.symbols[f.field_serializer_name_sym]
        else:
            self.serializer_name = None

        self.serializer_version = f.field_serializer_version

        if f.var_encoder_sym != 0:
            self.encoder = ser.symbols[f.var_encoder_sym]
        else:
            self.encoder = None

        self.encode_flags = f.encode_flags
        self.bit_count = f.bit_count
        self.low_value = f.low_value
        self.high_value = f.high_value
        self.field_type = None
        self.serializer = None
        self.value = None
        self.model = fieldModelEnum.fieldModelSimple

        self.decoder = None
        self.base_decoder = None
        self.child_decoder = None

        if self.send_node == "(root)":
            self.send_node = ""

    def set_model(self, model):
        self.model = model

        if model == fieldModelEnum.fieldModelFixedArray.value:
            if self.field_type.base_type == "float32":
                if self.encoder == "coord":
                    self.decoder = "floatCoordDecoder"
                elif self.encoder == "simtime":
                    self.decoder = "simulationTimeDecoder"
                elif self.encoder == "runeTimeDecoder":
                    self.decoder = "runeTimeDecoder"
                elif self.bit_count == None or self.bit_count <= 0 or self.bit_count >= 32:
                    self.decoder = "noscaleDecoder"
                else:
                    self.decoder = "QuantizedFloatDecoder"
            elif self.field_type.base_type == "CNetworkedQuantizedFloat":
                self.decoder = "QuantizedFloatDecoder"
            elif self.field_type.base_type == "Vector":
                if self.encoder == "normal":
                    self.decoder = "vectorNormalDecoder"
                else:
                    if self.encoder == "coord":
                        self.decoder = "floatCoordDecoder_3"
                    elif self.encoder == "simtime":
                        self.decoder = "simulationTimeDecoder_3"
                    elif self.encoder == "runetime":
                        self.decoder = "runeTimeDecoder_3"
                    elif self.bit_count == None or self.bit_count <= 0 or self.bit_count >=32:
                        self.decoder = "noscaleDecoder_3"
                    else:
                        self.decoder = "quantizedFactor_3"
            elif self.field_type.base_type == "Vector2D":
                if self.encoder == "coord":
                    self.decoder = "floatCoordDecoder_2"
                elif self.encoder == "simtime":
                    self.decoder = "simulationTimeDecoder_2"
                elif self.encoder == "runetime":
                    self.decoder = "runeTimeDecoder_2"
                elif self.bit_count == None or self.bit_count <= 0 or self.bit_count >=32:
                    self.decoder = "noscaleDecoder_2"
                else:
                    self.decoder = "quantizedFactor_2"
            elif self.field_type.base_type == "Vector4D":
                if self.encoder == "coord":
                    self.decoder = "floatCoordDecoder_4"
                elif self.encoder == "simtime":
                    self.decoder = "simulationTimeDecoder_4"
                elif self.encoder == "runetime":
                    self.decoder = "runeTimeDecoder_4"
                elif self.bit_count == None or self.bit_count <= 0 or self.bit_count >=32:
                    self.decoder = "noscaleDecoder_4"
                else:
                    self.decoder = "quantizedFactor_4"
            elif self.field_type.base_type == "uint64":
                if self.encoder == "fixed64":
                    self.decoder = "fixed64Decoder"
                else:
                    self.decoder = "unsigned64Decoder"
            elif self.field_type.base_type == "QAngle":
                if self.encoder == "qangle_pitch_yaw":
                    n = int(self.bit_count)
                    self.decoder = "QAngle_1"
                elif self.bit_count == 0 and self.bit_count != None:
                    n = int(self.bit_count)
                    self.decoder = "QAngle_2"
                else:
                    self.decoder = "QAngle_3"
            elif self.field_type.base_type == "CHandle":
                self.decoder = "unsignedDecoder"
            elif self.field_type.base_type == "CStrongHandle":
                if self.encoder == "fixed64":
                    self.decoder = "fixed64Decoder"
                else:
                    self.decoder = "unsigned64Decoder"
            elif self.field_type.base_type == "CEntityHandle":
                self.decoder = "unsignedDecoder"
            elif self.field_type.base_type in fieldTypeDecoders:
                self.decoder = fieldTypeDecoders[self.field_type.base_type]
            else:
                self.decoder = "defaultDecoder"
        elif model == fieldModelEnum.fieldModelFixedTable.value:
            self.base_decoder = "booleanDecoder"
        elif model == fieldModelEnum.fieldModelVariableArray.value:
            if self.field_type.generic_type == None:
                print("return")
                return -1

            self.base_decoder = "unsignedDecoder"
            if self.field_type.generic_type.base_type in fieldTypeDecoders:
                self.child_decoder = fieldTypeDecoders[self.field_type.generic_type.base_type]
            else:
                self.child_decoder = "defaultDecoder"
        elif model == fieldModelEnum.fieldModelVariableTable.value:
            self.base_decoder = "unsignedDecoder"
        elif model == fieldModelEnum.fieldModelSimple.value:
            if self.field_type.base_type == "float32":
                if self.encoder == "coord":
                    self.decoder = "floatCoordDecoder"
                elif self.encoder == "simtime":
                    self.decoder = "simulationTimeDecoder"
                elif self.encoder == "runeTimeDecoder":
                    self.decoder = "runeTimeDecoder"
                elif self.bit_count == None or self.bit_count <= 0 or self.bit_count >= 32:
                    self.decoder = "noscaleDecoder"
                else:
                    self.decoder = "QuantizedFloatDecoder"
            elif self.field_type.base_type == "CNetworkedQuantizedFloat":
                self.decoder = "QuantizedFloatDecoder"
            elif self.field_type.base_type == "Vector":
                if self.encoder == "normal":
                    self.decoder = "vectorNormalDecoder"
                else:
                    if self.encoder == "coord":
                        self.decoder = "floatCoordDecoder_3"
                    elif self.encoder == "simtime":
                        self.decoder = "simulationTimeDecoder_3"
                    elif self.encoder == "runetime":
                        self.decoder = "runeTimeDecoder_3"
                    elif self.bit_count == None or self.bit_count <= 0 or self.bit_count >=32:
                        self.decoder = "noscaleDecoder_3"
                    else:
                        self.decoder = "quantizedFactor_3"
            elif self.field_type.base_type == "Vector2D":
                if self.encoder == "coord":
                    self.decoder = "floatCoordDecoder_2"
                elif self.encoder == "simtime":
                    self.decoder = "simulationTimeDecoder_2"
                elif self.encoder == "runetime":
                    self.decoder = "runeTimeDecoder_2"
                elif self.bit_count == None or self.bit_count <= 0 or self.bit_count >=32:
                    self.decoder = "noscaleDecoder_2"
                else:
                    self.decoder = "quantizedFactor_2"
            elif self.field_type.base_type == "Vector4D":
                if self.encoder == "coord":
                    self.decoder = "floatCoordDecoder_4"
                elif self.encoder == "simtime":
                    self.decoder = "simulationTimeDecoder_4"
                elif self.encoder == "runetime":
                    self.decoder = "runeTimeDecoder_4"
                elif self.bit_count == None or self.bit_count <= 0 or self.bit_count >=32:
                    self.decoder = "noscaleDecoder_4"
                else:
                    self.decoder = "quantizedFactor_4"
            elif self.field_type.base_type == "uint64":
                if self.encoder == "fixed64":
                    self.decoder = "fixed64Decoder"
                else:
                    self.decoder = "unsigned64Decoder"
            elif self.field_type.base_type == "QAngle":
                if self.encoder == "qangle_pitch_yaw":
                    n = int(self.bit_count)
                    self.decoder = "QAngle_1"
                elif self.bit_count != 0 and self.bit_count != None:
                    n = int(self.bit_count)
                    self.decoder = "QAngle_2"
                else:
                    self.decoder = "QAngle_3"
            elif self.field_type.base_type == "CHandle":
                self.decoder = "unsignedDecoder"
            elif self.field_type.base_type == "CStrongHandle":
                if self.encoder == "fixed64":
                    self.decoder = "fixed64Decoder"
                else:
                    self.decoder = "unsigned64Decoder"
            elif self.field_type.base_type == "CEntityHandle":
                self.decoder = "unsignedDecoder"
            elif self.field_type.base_type in fieldTypeDecoders:
                self.decoder = fieldTypeDecoders[self.field_type.base_type]
            else:
                self.decoder = "defaultDecoder"


class QuantizedFloatDecoder:
    def __init__(self, bit_count, flags, low_value, high_value):
        self.low = 0
        self.high = 0
        self.high_low_mul = 0
        self.dec_mul = 0
        self.offset = 0
        self.bit_count = 0
        self.flags = 0
        self.noscale = False

        qff_rounddown = 1 << 0
        qff_roundup = 1 << 1
        qff_encode_zero = 1 << 2
        qff_encode_integers = 1 << 3

        if bit_count == 0 or bit_count >= 32:
            self.noscale = True
            self.bit_count = 32
        else:
            self.noscale = False
            self.bit_count = bit_count
            self.offset = 0.0

            if low_value != 0.0:
                self.low = low_value
            else:
                self.low = 0.0

            if high_value != 0.0:
                self.high = high_value
            else:
                self.high = 1.0

        if flags != 0:
            self.flags = flags
        else:
            self.flags = 0

        self.validate_flags()

        steps = 1 << bit_count

        range_value = 0.0
        if self.flags & qff_rounddown != 0:
            range_value = self.high - self.low
            self.offset = range_value / float(steps)
            self.high -= self.offset
        elif self.flags & qff_roundup != 0:
            range_value = self.high - self.low
            self.offset = range_value / float(steps)
            self.low += self.offset

        if (self.flags & qff_encode_integers) != 0:
            delta = self.high - self.low

            if delta < 1:
                delta = 1

            delta_log2 = math.ceil(math.log(float(delta), 2))
            range_value2 = 1 << int(delta_log2)
            bc = self.bit_count

            while True:
                if (1 << int(bc)) > range_value2:
                    break
                else:
                    bc += 1

            if bc > self.bit_count:
                self.bit_count = bc
                steps = 1 << int(self.bit_count)

            self.offset = float(range_value2) / float(steps)
            self.high = self.low + float(range_value2) - self.offset

        self.high_low_mul = None
        self.dec_mul = None

        self.assign_multipliers(int(steps))

        #print("self.flags & qff_rounddown: ", self.flags & qff_rounddown)
        #print("self.quantize(self.high): ", self.quantize(self.high))
        if self.flags & qff_rounddown != 0:
            if self.quantize(self.low) == self.low:
                self.flags &= ~qff_rounddown

        if self.flags & qff_roundup != 0:
            if self.quantize(self.high) == self.high:
                self.flags &= ~qff_roundup

        if self.flags & qff_encode_zero != 0:
            if self.quantize(0.0) == self.high:
                self.flags &= ~qff_encode_zero

    def quantize(self, val):
        if val < self.low:
            if self.flags & qff_roundup == 0:
                print("Field tried to quantize an out of range value")
                return

            return self.low
        elif val > self.high:
            if self.flags & qff_rounddown == 0:
                print("Field tried to quantize an out of range value")
                return

            return self.high

        #self.high_low_mul = 512
        i = int((val - self.low) * self.high_low_mul)

        return self.low + (self.high - self.low) * (float(i) * self.dec_mul)

    def assign_multipliers(self, steps):
        self.high_low_mul = 0.0
        range_value = self.high - self.low

        high = 0
        if self.bit_count == 32:
            high = 0xFFFFFFFE
        else:
            high = (1 << self.bit_count) - 1

        high_mul = float(0.0)
        if abs(float(range_value)) <= 0.0:
            high_mul = float(high)
        else:
            high_mul = float(high) / range_value

        if (high_mul * range_value > float(high)) or (float(high_mul - range_value) > float(high)):
            multipliers = [0.9999, 0.99, 0.9, 0.8, 0.7]

            for mult in multipliers:
                high_mul = float(high) / range_value * mult

                if (high_mul * range_value > float(high)) or (float(high_mul * range_value) > float(high)):
                    continue

                break

        self.high_low_mul = high_mul
        self.dec_mul = 1.0 / float(steps - 1)

        if self.high_low_mul == 0.0:
            print("Error computing high / low multiplier")
            return -1

    def validate_flags(self):
        qff_rounddown = 1 << 0
        qff_roundup = 1 << 1
        qff_encode_zero = 1 << 2
        qff_encode_integers = 1 << 3

        if self.flags == 0:
            return -1

        if self.low == 0.0 and (self.flags & qff_rounddown != 0) or (self.high == 0.0 and (self.flags & qff_roundup) != 0):
            self.flags &= ~qff_encode_zero

        if self.low == 0.0 and (self.flags & qff_encode_zero != 0):
            self.flags |= ~qff_rounddown
            self.flags &= ~qff_encode_zero

        if self.high == 0.0 and (self.flags & qff_encode_zero != 0):
            self.flags |= ~qff_roundup
            self.flags &= ~qff_encode_zero

        if self.low == 0.0 or self.high < 0.0:
            self.flags &= ~qff_encode_zero

        if self.flags & qff_encode_integers != 0:
            self.flags &= ~(qff_roundup | qff_rounddown | qff_encode_zero)

        if self.flags & (qff_rounddown | qff_roundup) == (qff_rounddown | qff_roundup):
            return -1

    def decode(self, r):
        qff_rounddown = 1 << 0
        qff_roundup = 1 << 1
        qff_encode_zero = 1 << 2
        qff_encode_integers = 1 << 3

        if ((self.flags & qff_rounddown) != 0) and r.read_boolean():
            return self.low
        elif ((self.flags & qff_roundup) != 0) and r.read_boolean():
            return self.high
        elif ((self.flags & qff_encode_zero)) != 0 and r.read_boolean():
            return 0.0
        else:
            return self.low + (self.high - self.low) * float(r.read_bits(self.bit_count)) * self.dec_mul


class DemoClass:
    def __init__(self, class_id, name, serializer):
        self.class_id = class_id
        self.name = name
        self.serializer = serializer


class FieldState:
    def __init__(self):
        self.state = [None] * 8

    def set(self, fp, v):
        z = 0
        for i in range(fp.last + 1):
            z = fp.path[i]
            y = len(self.state)
            if y < z + 2:
                k = [None] * max(z+2, y*2)
                self.state = copy.deepcopy(k)
                self.state = k

            if i == fp.last:
                if self.state[z] == None:
                    self.state[z] = v

                return

            if self.state[z] == None:
                self.state[z] = FieldState()


def SetFieldState(s, fp, v):
    dummy_field_state = FieldState()

    x = s
    z = 0
    for i in range(fp.last + 1):
        z = fp.path[i]
        y = len(x.state)
        if y < z + 2:
            k = [None] * max(z+2, y*2)
            for j in range(0, len(x.state)):
                k[j] = copy.deepcopy(x.state[j])

            x.state = k

        if i == fp.last:
            if type(x.state[z]) != type(dummy_field_state):
                x.state[z] = v

            return

        if type(x.state[z]) != type(dummy_field_state):
            x.state[z] = FieldState()

        x = x.state[z]


def GetFieldState(s, fp):
    dummy_field_state = FieldState()

    x = s
    z = 0
    for i in range(fp.last + 1):
        z = fp.path[i]

        if len(x.state) < z + 2:
            return None

        if i == fp.last:
            return x.state[z]

        if type(x.state[z]) != type(dummy_field_state):
            return None

        x = x.state[z]

    #time.sleep(1)

    return None


def GetFieldPathsFromSerializer(s, fp, state):
    serializer = s

    results = []
    for i, f in enumerate(serializer.fields):
        fp.path[fp.last] = i

        result = GetFieldPathsFromField(f, fp, state)
        if len(result) != 0:
            if type(result[0]) != list:
                x_name = [path.path for path in result]

                for value in result:
                    results.append(value)
            else:
                x_name = [path.path for path in result[0]]

                for value in result[0]:
                    results.append(value)
        
    return results


def GetFieldPathsFromField(f, fp, state):
    dummy_field_state = FieldState()
    x = []

    if f.model == fieldModelEnum.fieldModelFixedArray.value:
        sub = GetFieldState(state, fp)
        if type(sub) == type(dummy_field_state):
            fp.last += 1
            for i, v in enumerate(sub.state):
                if v != None:
                    fp.path[fp.last] = i
                    x.append(copy.deepcopy(fp))

            fp.last -= 1
    elif f.model == fieldModelEnum.fieldModelFixedTable.value:
        sub = GetFieldState(state, fp)
        if type(sub) == type(dummy_field_state):
            fp.last += 1
            serializer_of_field = f.serializer
            x.append(GetFieldPathsFromSerializer(serializer_of_field, fp, sub))
            fp.last -= 1
    elif f.model == fieldModelEnum.fieldModelVariableArray.value:
        sub = GetFieldState(state, fp)
        if type(sub) == type(dummy_field_state):
            fp.last += 1
            for i, v in enumerate(sub.state):
                if v != None:
                    fp.path[fp.last] = i
                    x.append(copy.deepcopy(fp))

            fp.last -= 1
    elif f.model == fieldModelEnum.fieldModelVariableTable.value:
        sub = GetFieldState(state, fp)
        if type(sub) == type(dummy_field_state):
            fp.last += 2
            for i, v in enumerate(sub.state):
                if type(v) == type(dummy_field_state):
                    fp.path[fp.last - 1] = i
                    serializer_of_field = f.serializer
                    x.append(GetFieldPathsFromSerializer(serializer_of_field, fp, v))

            fp.last -= 2
    elif f.model == fieldModelEnum.fieldModelSimple.value:
        x.append(copy.deepcopy(fp))

    return x


def GetNameForFieldPathFromSerializer(s, fp, pos):
    field_of_serializer = s.fields[fp.path[pos]]
    result = GetNameForFieldPathFromField(field_of_serializer, fp, pos + 1)
    result = "".join(result)

    return result


def GetNameForFieldPathFromField(f, fp, pos):
    x = [f.var_name]

    if f.model == fieldModelEnum.fieldModelFixedArray.value:
        if fp.last == pos:
            result = "%04d" % (fp.path[pos])
            x.append(result)
    elif f.model == fieldModelEnum.fieldModelFixedTable.value:
        if fp.last >= pos:
            serializer_of_field = f.serializer
            result = GetNameForFieldPathFromSerializer(serializer_of_field, fp, pos)
            x.append(result)
    elif f.model == fieldModelEnum.fieldModelVariableArray.value:
        if fp.last == pos:
            result = "%04d" % (fp.path[pos])
            x.append(result)
    elif f.model == fieldModelEnum.fieldModelVariableTable.value:
        if fp.last != pos - 1:
            result = "%04d" % (fp.path[pos])
            x.append(result)

            if fp.last != pos:
                serializer_of_field = f.serializer
                result = GetNameForFieldPathFromSerializer(serializer_of_field, fp, pos + 1)
                x.append(result)

    return x


def EntityMap(entity):
    demo_class = entity.demo_class
    serializer = demo_class.serializer

    fp = FieldPath()
    values = {}
    field_paths = GetFieldPathsFromSerializer(serializer, fp, entity.state)

    m_iPlayerID = None
    m_cellX = None
    for field_path in field_paths:
        entity_result = GetNameForFieldPathFromSerializer(serializer, field_path, 0)
        value = GetFieldState(entity.state, field_path)
        if value == None:
            value = "None"

        if entity_result == 'm_iPlayerID':
            #print("value: ", value)
            m_iPlayerID = value

        if entity_result == 'CBodyComponentm_cellX':
            m_cellX = value

        values[entity_result] = GetFieldState(entity.state, field_path)

    return values


def ParseName(name):
    name_list = name.split('_')[3:]
    #name = ''.join(name_list)
    #name_split = re.sub( r"([A-Z])", r" \1", name).split()
    #for i in range(len(name_split)):
    #    name_split[i] = name_split[i].lower()

    name = "_".join(name_list)

    return name


def GetHeroName(demo_class):
    demo_class_name = demo_class.name
    result = demo_class_name.startswith('CDOTA_Unit_Hero')
    if result == True:
        hero_name_list = demo_class_name.split('_')[3:]
        hero_name = ''.join(hero_name_list)
        hero_name_split = re.sub( r"([A-Z])", r" \1", hero_name).split()
        for i in range(len(hero_name_split)):
            hero_name_split[i] = hero_name_split[i].lower()

        hero_name = "_".join(hero_name_split)

        return hero_name
    else:
        return None


def GetNpcName(demo_class):
    demo_class_name = demo_class.name
    result = demo_class_name.startswith('CDOTA_BaseNPC')
    if result == True:
        npc_name_list = demo_class_name.split('_')[2:]
        npc_name = ''.join(npc_name_list)
        npc_name_split = re.sub( r"([A-Z])", r" \1", npc_name).split()
        for i in range(len(npc_name_split)):
            npc_name_split[i] = npc_name_split[i].lower()

        npc_name = "_".join(npc_name_split)

        return npc_name
    else:
        return None


def GetItemInfo(item_handle, entities, entity_names_string_table):
    handle_mask = (1 << 14) - 1

    item_name = None
    item_cool = 0
    item_num = 0
    if item_handle != 16777215:
        item_handle &= handle_mask

        if item_handle in entities:
            item_entity = entities[item_handle]
            if item_entity != None:
                item_E_Map = EntityMap(item_entity)

                if item_E_Map["m_pEntitym_nameStringableIndex"] != -1:
                    m_pEntitym_nameStringableIndex = item_E_Map["m_pEntitym_nameStringableIndex"]
                    item_name = entity_names_string_table.items[m_pEntitym_nameStringableIndex]
                    item_name = item_name.key
                    item_name = item_name.split('_')[1:]
                    item_name = "_".join(item_name)
                    #print("item_name: ", item_name)
                    if item_name.split('_')[0] == "recipe":
                        item_name = "recipe"

                    item_cool = item_E_Map["m_fCooldown"]
                    item_num = item_E_Map["m_iInitialCharges"]

    return item_name, item_cool, item_num


def GetItemsInfo(entity_info, entities, entity_names_string_table):
    #print("entity_info['m_hItems0003']: ", entity_info['m_hItems0003'])
    item_name_0, item_cool_0, item_num_0 = GetItemInfo(entity_info['m_hItems0000'], entities, entity_names_string_table)
    item_name_1, item_cool_1, item_num_1 = GetItemInfo(entity_info['m_hItems0001'], entities, entity_names_string_table)
    item_name_2, item_cool_2, item_num_2 = GetItemInfo(entity_info['m_hItems0002'], entities, entity_names_string_table)
    item_name_3, item_cool_3, item_num_3 = GetItemInfo(entity_info['m_hItems0003'], entities, entity_names_string_table)
    item_name_4, item_cool_4, item_num_4 = GetItemInfo(entity_info['m_hItems0004'], entities, entity_names_string_table)
    item_name_5, item_cool_5, item_num_5 = GetItemInfo(entity_info['m_hItems0005'], entities, entity_names_string_table)

    return {
            "item0": {"item_name": item_name_0, "item_cool": item_cool_0, "item_num": item_num_0},
            "item1": {"item_name": item_name_1, "item_cool": item_cool_1, "item_num": item_num_1},
            "item2": {"item_name": item_name_2, "item_cool": item_cool_2, "item_num": item_num_2},
            "item3": {"item_name": item_name_3, "item_cool": item_cool_3, "item_num": item_num_3},
            "item4": {"item_name": item_name_4, "item_cool": item_cool_4, "item_num": item_num_4},
            "item5": {"item_name": item_name_5, "item_cool": item_cool_5, "item_num": item_num_5}
           }


def GetAbilityInfo(ability_handle, entities, entity_names_string_table):
    handle_mask = (1 << 14) - 1

    ability_name = None
    ability_cool = 0
    ability_level = 0
    if ability_handle != 16777215:
        ability_handle &= handle_mask

        ability_entity = entities[ability_handle]
        if ability_entity != None:
            ability_E_Map = EntityMap(ability_entity)
            #print("ability_E_Map: ", ability_E_Map)

            if ability_E_Map["m_pEntitym_nameStringableIndex"] != -1:
                m_pEntitym_nameStringableIndex = ability_E_Map["m_pEntitym_nameStringableIndex"]
                ability_name = entity_names_string_table.items[m_pEntitym_nameStringableIndex]
                ability_name = ability_name.key

                ability_cool = ability_E_Map["m_fCooldown"]
                ability_level = ability_E_Map["m_iLevel"]
                #print("ability_name: ", ability_name)
                #print("ability_E_Map: ", ability_E_Map)

    return ability_name, ability_cool, ability_level


def GetAbilitesInfo(entity_info, entities, entity_names_string_table):
    #print("entity_info['m_hAbilities0003']: ", entity_info['m_hAbilities0003'])
    ability_name_0, ability_cool_0, ability_level_0 = GetAbilityInfo(entity_info['m_hAbilities0000'], entities, entity_names_string_table)
    ability_name_1, ability_cool_1, ability_level_1 = GetAbilityInfo(entity_info['m_hAbilities0001'], entities, entity_names_string_table)
    ability_name_2, ability_cool_2, ability_level_2 = GetAbilityInfo(entity_info['m_hAbilities0002'], entities, entity_names_string_table)
    ability_name_3, ability_cool_3, ability_level_3 = GetAbilityInfo(entity_info['m_hAbilities0003'], entities, entity_names_string_table)
    ability_name_4, ability_cool_4, ability_level_4 = GetAbilityInfo(entity_info['m_hAbilities0004'], entities, entity_names_string_table)
    ability_name_5, ability_cool_5, ability_level_5 = GetAbilityInfo(entity_info['m_hAbilities0005'], entities, entity_names_string_table)

    return {
            "ability0": {"ability_name": ability_name_0, "ability_cool": ability_cool_0, "ability_level": ability_level_0},
            "ability1": {"ability_name": ability_name_1, "ability_cool": ability_cool_1, "ability_level": ability_level_1},
            "ability2": {"ability_name": ability_name_2, "ability_cool": ability_cool_2, "ability_level": ability_level_2},
            "ability3": {"ability_name": ability_name_3, "ability_cool": ability_cool_3, "ability_level": ability_level_3},
            "ability4": {"ability_name": ability_name_4, "ability_cool": ability_cool_4, "ability_level": ability_level_4},
            "ability5": {"ability_name": ability_name_5, "ability_cool": ability_cool_5, "ability_level": ability_level_5}
           }


MAP_SIZE = 32768
MAP_HALF_SIZE = MAP_SIZE / 2
CELL_SIZE = 128
def GetHeroInfo(entity):
    E_Map = EntityMap(entity)
    #print("E_Map['m_hModifierParent']: ", E_Map['m_hModifierParent'])

    #for key in E_Map:
    #    print("key: ", key)
    #    print("E_Map[key]: ", E_Map[key])
    #print("")

    hero_name = GetHeroName(entity.demo_class)

    m_iPlayerID = None
    m_cellX = None
    m_cellY = None
    m_vecX = None
    m_vecY = None
    m_iCurrentXP = None
    m_iCurrentLevel = None
    m_iTeamNum = None
    angRotation = None 

    m_hAbilities0000 = None
    m_hAbilities0001 = None
    m_hAbilities0002 = None
    m_hAbilities0003 = None
    m_hAbilities0004 = None
    m_hAbilities0005 = None

    m_hItems0000 = None
    m_hItems0001 = None
    m_hItems0002 = None
    m_hItems0003 = None
    m_hItems0004 = None
    m_hItems0005 = None

    m_bIsWaitingToSpawn = None

    if 'm_iPlayerID' in E_Map:
        m_iPlayerID = E_Map["m_iPlayerID"]

    if 'CBodyComponentm_cellX' in E_Map:
        m_cellX = E_Map["CBodyComponentm_cellX"]
        m_cellY = E_Map["CBodyComponentm_cellY"]

    if 'CBodyComponentm_vecX' in E_Map:
        m_vecX = E_Map["CBodyComponentm_vecX"]
        m_vecY = E_Map["CBodyComponentm_vecY"]

    if 'CBodyComponentm_angRotation' in E_Map:
        angRotation = E_Map["CBodyComponentm_angRotation"][1]

    if 'm_iMaxHealth' in E_Map:
        m_iHealth = E_Map["m_iHealth"]
        m_iMaxHealth = E_Map["m_iMaxHealth"]

    if 'm_iCurrentXP' in E_Map:
        m_iCurrentXP = E_Map["m_iCurrentXP"]

    if 'm_iCurrentLevel' in E_Map:
        m_iCurrentLevel = E_Map["m_iCurrentLevel"]

    if 'm_iTeamNum' in E_Map:
        m_iTeamNum = E_Map["m_iTeamNum"]

    if 'm_flMaxMana' in E_Map:
        m_flMana = E_Map["m_flMana"]
        m_flMaxMana = E_Map["m_flMaxMana"]

    if 'm_hAbilities0000' in E_Map:
        m_hAbilities0000 = E_Map["m_hAbilities0000"]
        m_hAbilities0001 = E_Map["m_hAbilities0001"]
        m_hAbilities0002 = E_Map["m_hAbilities0002"]
        m_hAbilities0003 = E_Map["m_hAbilities0003"]
        m_hAbilities0004 = E_Map["m_hAbilities0004"]
        m_hAbilities0005 = E_Map["m_hAbilities0005"]

    if 'm_hItems0000' in E_Map:
        m_hItems0000 = E_Map["m_hItems0000"]
        m_hItems0001 = E_Map["m_hItems0001"]
        m_hItems0002 = E_Map["m_hItems0002"]
        m_hItems0003 = E_Map["m_hItems0003"]
        m_hItems0004 = E_Map["m_hItems0004"]
        m_hItems0005 = E_Map["m_hItems0005"]

    if 'm_bIsWaitingToSpawn' in E_Map:
        m_bIsWaitingToSpawn = E_Map["m_bIsWaitingToSpawn"]

    x_temp = m_cellX * CELL_SIZE + m_vecX
    y_temp = m_cellY * CELL_SIZE + m_vecY
    hero_location_x = -x_temp + (MAP_HALF_SIZE / 2)
    hero_location_y = y_temp - (MAP_HALF_SIZE / 2)

    info = {"m_iPlayerID": m_iPlayerID,
            "hero_location_x": hero_location_x / 4.0,
            "hero_location_y": hero_location_y / 4.0,
            "angRotation": angRotation,
            "m_iCurrentXP": m_iCurrentXP,
            "m_iCurrentLevel": m_iCurrentLevel,
            "hero_name": hero_name,
            "m_iHealth": m_iHealth,
            "m_iMaxHealth": m_iMaxHealth,
            "m_iTeamNum": m_iTeamNum,
            "m_flMana": m_flMana,
            "m_flMaxMana": m_flMaxMana,
            "m_hAbilities0000": m_hAbilities0000,
            "m_hAbilities0001": m_hAbilities0001,
            "m_hAbilities0002": m_hAbilities0002,
            "m_hAbilities0003": m_hAbilities0003,
            "m_hAbilities0004": m_hAbilities0004,
            "m_hAbilities0005": m_hAbilities0005,
            "m_hItems0000": m_hItems0000,
            "m_hItems0001": m_hItems0001,
            "m_hItems0002": m_hItems0002,
            "m_hItems0003": m_hItems0003,
            "m_hItems0004": m_hItems0004,
            "m_hItems0005": m_hItems0005,
            "m_bIsWaitingToSpawn": m_bIsWaitingToSpawn
           }

    return info


def GetNpcInfo(entity):
    E_Map = EntityMap(entity)

    npc_name = GetNpcName(entity.demo_class)

    m_nEntityId = None
    m_cellX = None
    m_cellY = None
    m_vecX = None
    m_vecY = None
    m_iTeamNum = None
    m_hModel = None

    handle_mask = (1 << 14) - 1

    if 'm_nEntityId' in E_Map:
        m_nEntityId = E_Map["m_nEntityId"]
        m_nEntityId &= handle_mask 

    if 'CBodyComponentm_cellX' in E_Map:
        m_cellX = E_Map["CBodyComponentm_cellX"]
        m_cellY = E_Map["CBodyComponentm_cellY"]

    if 'CBodyComponentm_vecX' in E_Map:
        m_vecX = E_Map["CBodyComponentm_vecX"]
        m_vecY = E_Map["CBodyComponentm_vecY"]

    if 'm_iMaxHealth' in E_Map:
        m_iHealth = E_Map["m_iHealth"]
        m_iMaxHealth = E_Map["m_iMaxHealth"]

    if 'm_iTeamNum' in E_Map:
        m_iTeamNum = E_Map["m_iTeamNum"]

    if 'CBodyComponentm_hModel' in E_Map:
        m_hModel = E_Map["CBodyComponentm_hModel"]
        m_hModel &= handle_mask 

    x_temp = m_cellX * CELL_SIZE + m_vecX
    y_temp = m_cellY * CELL_SIZE + m_vecY
    location_x = -x_temp + (MAP_HALF_SIZE / 2)
    location_y = y_temp - (MAP_HALF_SIZE / 2)

    info = {
            "npc_name": npc_name,
            "m_nEntityId": m_nEntityId,
            "location_x": location_x / 4.0,
            "location_y": location_y / 4.0,
            "m_iHealth": m_iHealth,
            "m_iMaxHealth": m_iMaxHealth,
            "m_iTeamNum": m_iTeamNum,
            "m_hModel": m_hModel
           }

    return info


def GetRuneInfo(entity, entity_names_string_table):
    E_Map = EntityMap(entity)
    #for key in E_Map:
      #print("key: ", key)
      #print("E_Map[key]: ", E_Map[key])
    #print("")

    name = None
    m_nEntityId = None
    m_cellX = None
    m_cellY = None
    m_vecX = None
    m_vecY = None
    m_iTeamNum = None
    m_pEntitym_nameStringableIndex = None

    handle_mask = (1 << 14) - 1

    if 'm_nEntityId' in E_Map:
        m_nEntityId = E_Map["m_nEntityId"]
        m_nEntityId &= handle_mask 

    if 'CBodyComponentm_cellX' in E_Map:
        m_cellX = E_Map["CBodyComponentm_cellX"]
        m_cellY = E_Map["CBodyComponentm_cellY"]

    if 'CBodyComponentm_vecX' in E_Map:
        m_vecX = E_Map["CBodyComponentm_vecX"]
        m_vecY = E_Map["CBodyComponentm_vecY"]

    if 'm_pEntitym_nameStringableIndex' in E_Map:
        m_pEntitym_nameStringableIndex = E_Map["m_pEntitym_nameStringableIndex"]
        #print("m_pEntitym_nameStringableIndex: ", m_pEntitym_nameStringableIndex)
        if m_pEntitym_nameStringableIndex != -1:
            name = entity_names_string_table.items[m_pEntitym_nameStringableIndex]
            #print("name: ", name)

    x_temp = m_cellX * CELL_SIZE + m_vecX
    y_temp = m_cellY * CELL_SIZE + m_vecY
    location_x = -x_temp + (MAP_HALF_SIZE / 2)
    location_y = y_temp - (MAP_HALF_SIZE / 2)

    info = {"m_nEntityId": m_nEntityId,
            "location_x": location_x / 4.0,
            "location_y": location_y / 4.0,
            "name": name
           }

    return info


class NewEntity:
    def __init__(self, index, serial, demo_class):
        self.index = index
        self.serial = serial
        self.demo_class = demo_class
        self.active = True
        self.state = FieldState()
        self.fpCache = {}
        self.fpNoop = {}


class StringTable:
    def __init__(self, index, name, user_data_fixed_size, user_data_size, user_data_size_bits, flags):
        self.index = index
        self.name = name
        self.items = {}
        self.user_data_fixed_size = user_data_fixed_size
        self.user_data_size = user_data_size
        self.user_data_size_bits = user_data_size_bits
        self.flags = flags


class StringTableItem:
    def __init__(self, index, key, value):
        self.index = index
        self.key = key
        self.value = value


def ParseStringTable(buf, num_updates, name, user_data_fixed_size, user_data_size, user_data_size_bits, flags):
    items = []

    r = FileReader(BytesIO(buf))
    index = -1
    keys = []
    if len(buf) != 0:
        for i in range(0, num_updates):
            key = ""
            value = []

            incr = r.read_boolean()
            if incr == True:
                index += 1
            else:
                index = r.read_var_uint32() + 1

            has_key = r.read_boolean()
            if has_key:
                useHistory = r.read_boolean()
                if useHistory:
                    pos = r.read_bits(5)
                    size = r.read_bits(5)

                    if int(pos) >= len(keys):
                        read_string = r.read_string().decode('utf-8')
                        key += read_string
                    else:
                        s = keys[pos]

                        if int(size) > len(s):
                            read_string = r.read_string().decode('utf-8')
                            key += s + read_string
                        else:
                            read_string = r.read_string().decode('utf-8')
                            key += s[0:size] + read_string
                else:
                    read_string = r.read_string().decode('utf-8')
                    key = read_string
                
                stringtableKeyHistorySize = 32

                if len(keys) >= stringtableKeyHistorySize:
                    for k in range(0, len(keys) - 1):
                        keys[k] = copy.deepcopy(keys[k+1])    

                    keys[len(keys)-1] = ""
                    keys = keys[:len(keys)-1]

                keys.append(key)


            has_value = r.read_boolean()
            if has_value:
                bit_size = 0
                if user_data_fixed_size:
                    bit_size = user_data_size_bits
                else:
                    if flags & 0x1 != 0:
                        value = r.readBoolean()
                    
                    bit_size = r.read_bits(17) * 8

                value = r.read_bits_as_bytes(bit_size)

            #print("key: ", key)
            #print("value: ", value)
            new_item = StringTableItem(index, key, value)
            items.append(new_item)

        #print("")


    return items


class FieldPatch:
    def __init__(self, min_build, max_build, patch):
        self.min_build = min_build
        self.max_build = max_build
        self.patch = patch

    def should_apply(self, build):
        if self.min_build == 0 and self.max_build == 0:
            return True

        return (build >= self.min_build) and (build <= self.max_build)


def field_patch_func_1(f):
    case_list_1 = ["angExtraLocalAngles", "angLocalAngles", "m_angInitialAngles",
                   "m_angRotation", "m_ragAngles", "m_vLightDirection"]

    case_list_2 = ["dirPrimary", "localSound", "m_flElasticity", "m_location",
                   "m_poolOrigin", "m_ragPos", "m_vecEndPos", "m_vecLadderDir",
                   "m_vecPlayerMountPositionBottom", "m_vecPlayerMountPositionTop",
                   "m_viewtarget", "m_WorldMaxs", "m_WorldMins", "origin", "vecLocalOrigin"]

    if f.var_name in case_list_1:
        if f.parent_name == "CBodyComponentBaseAnimatingOverlay":
            f.encoder = "qangle_pitch_yaw"
        else:
            f.encoder = "QAngle"
    elif f.var_name in case_list_2:
        f.encoder = "normal"


def field_patch_func_2(f):
    if f.var_name in ["m_flMana", "m_flMaxMana"]:
        f.low_value = None
        f.high_value = 8192


def field_patch_func_3(f):
    if f.var_name in ["m_bItemWhiteList", "m_bWorldTreeState", "m_iPlayerIDsInControl", "m_iPlayerSteamID",
                      "m_ulTeamBannerLogo", "m_ulTeamBaseLogo", "m_ulTeamLogo"]:
        f.encoder = "fixed64"


def field_patch_func_4(f):
    if f.var_name in ["m_flSimulationTime", "m_flAnimTime"]:
        f.encoder = "simtime"
    elif f.var_name in ["m_flRuneTime"]:
        f.encoder = "runetime"


FieldPatches = [
                FieldPatch(0, 990, field_patch_func_1),
                FieldPatch(0, 954, field_patch_func_2),
                FieldPatch(1016, 1027, field_patch_func_3),
                FieldPatch(0, 0, field_patch_func_4),
                ]


class PendingMessage:
    def __init__(self, tick, cmd, message):
        self.tick = tick
        self.cmd = cmd
        self.message = message


class FieldPath:
    def __init__(self):
        self.path = [-1, 0, 0, 0, 0, 0]
        self.last = 0
        self.done = False


class FieldPathOp:
    def __init__(self, name, weight, fn):
        self.name = name
        self.weight = weight
        self.fn = fn 


class HuffmanLeaf:
    def __init__(self, weight, value):
        self.weight = weight
        self.value = value


def PlusOne(r, fp):
    fp.path[fp.last] += 1


def PlusTwo(r, fp):
    fp.path[fp.last] += 2


def PlusThree(r, fp):
    fp.path[fp.last] += 3


def PlusFour(r, fp):
    fp.path[fp.last] += 4


def PlusN(r, fp):
    fp.path[fp.last] += (int(r.read_ubit_var_field_path()) + 5)


def PushOneLeftDeltaZeroRightZero(r, fp):
    fp.last += 1
    fp.path[fp.last] = 0


def PushOneLeftDeltaZeroRightNonZero(r, fp):
    fp.last += 1
    fp.path[fp.last] = int(r.read_ubit_var_field_path())


def PushOneLeftDeltaOneRightZero(r, fp):
    fp.path[fp.last] += 1
    fp.last += 1
    fp.path[fp.last] = 0


def PushOneLeftDeltaOneRightNonZero(r, fp):
    fp.path[fp.last] += 1
    fp.last += 1
    fp.path[fp.last] = int(r.read_ubit_var_field_path())


def PushOneLeftDeltaNRightZero(r, fp):
    fp.path[fp.last] += int(r.read_ubit_var_field_path())
    fp.last += 1
    fp.path[fp.last] = 0


def PushOneLeftDeltaNRightNonZero(r, fp):
    fp.path[fp.last] += int(r.read_ubit_var_field_path()) + 2
    fp.last += 1
    fp.path[fp.last] = int(r.read_ubit_var_field_path()) + 1


def PushOneLeftDeltaNRightNonZeroPack6Bits(r, fp):
    fp.path[fp.last] += int(r.read_bits(3)) + 2
    fp.last += 1
    fp.path[fp.last] = int(r.read_bits(3)) + 1


def PushOneLeftDeltaNRightNonZeroPack8Bits(r, fp):
    fp.path[fp.last] += int(r.read_bits(4)) + 2
    fp.last += 1
    fp.path[fp.last] = int(r.read_bits(4)) + 1


def PushTwoLeftDeltaZero(r, fp):
    fp.last += 1
    fp.path[fp.last] += int(r.read_ubit_var_field_path())
    fp.last += 1
    fp.path[fp.last] += int(r.read_ubit_var_field_path())


def PushTwoPack5LeftDeltaZero(r, fp):
    fp.last += 1
    fp.path[fp.last] = int(r.read_bits(5))
    fp.last += 1
    fp.path[fp.last] = int(r.read_bits(5))


def PushThreeLeftDeltaZero(r, fp):
    fp.last += 1
    fp.path[fp.last] += int(r.read_ubit_var_field_path())
    fp.last += 1
    fp.path[fp.last] += int(r.read_ubit_var_field_path())
    fp.last += 1
    fp.path[fp.last] += int(r.read_ubit_var_field_path())


def PushThreePack5LeftDeltaZero(r, fp):
    fp.last += 1
    fp.path[fp.last] = int(r.read_bits(5))
    fp.last += 1
    fp.path[fp.last] = int(r.read_bits(5))
    fp.last += 1
    fp.path[fp.last] = int(r.read_bits(5))


def PushTwoLeftDeltaOne(r, fp):
    fp.path[fp.last] += 1
    fp.last += 1
    fp.path[fp.last] += int(r.read_ubit_var_field_path())
    fp.last += 1
    fp.path[fp.last] += int(r.read_ubit_var_field_path())


def PushTwoPack5LeftDeltaOne(r, fp):
    fp.path[fp.last] += 1
    fp.last += 1
    fp.path[fp.last] += int(r.read_bits(5))
    fp.last += 1
    fp.path[fp.last] += int(r.read_bits(5))


def PushThreeLeftDeltaOne(r, fp):
    fp.path[fp.last] += 1
    fp.last += 1
    fp.path[fp.last] += int(r.read_ubit_var_field_path())
    fp.last += 1
    fp.path[fp.last] += int(r.read_ubit_var_field_path())
    fp.last += 1
    fp.path[fp.last] += int(r.read_ubit_var_field_path())


def PushThreePack5LeftDeltaOne(r, fp):
    fp.path[fp.last] += 1
    fp.last += 1
    fp.path[fp.last] += int(r.read_bits(5))
    fp.last += 1
    fp.path[fp.last] += int(r.read_bits(5))
    fp.last += 1
    fp.path[fp.last] += int(r.read_bits(5))


def PushTwoLeftDeltaN(r, fp):
    fp.path[fp.last] += int(r.read_ubit_var()) + 2
    fp.last += 1
    fp.path[fp.last] += int(r.read_ubit_var_field_path())
    fp.last += 1
    fp.path[fp.last] += int(r.read_ubit_var_field_path())


def PushTwoPack5LeftDeltaN(r, fp):
    fp.path[fp.last] += int(r.read_ubit_var()) + 2
    fp.last += 1
    fp.path[fp.last] += int(r.read_bits(5))
    fp.last += 1
    fp.path[fp.last] += int(r.read_bits(5))


def PushThreeLeftDeltaN(r, fp):
    fp.path[fp.last] += int(r.read_ubit_var()) + 2
    fp.last += 1
    fp.path[fp.last] += int(r.read_ubit_var_field_path())
    fp.last += 1
    fp.path[fp.last] += int(r.read_ubit_var_field_path())
    fp.last += 1
    fp.path[fp.last] += int(r.read_ubit_var_field_path())


def PushThreePack5LeftDeltaN(r, fp):
    fp.path[fp.last] += int(r.read_ubit_var()) + 2
    fp.last += 1
    fp.path[fp.last] += int(r.read_bits(5))
    fp.last += 1
    fp.path[fp.last] += int(r.read_bits(5))
    fp.last += 1
    fp.path[fp.last] += int(r.read_bits(5))


def PushN(r, fp):
    n = int(r.read_ubit_var())
    fp.path[fp.last] += int(r.read_ubit_var())
    for i in range(0, n):
        fp.last += 1
        fp.path[fp.last] += int(r.read_ubit_var_field_path())


def PushNAndNonTopological(r, fp):
    for i in range(0, fp.last + 1):
        if r.read_boolean():
            fp.path[i] += int(r.read_var_int32()) + 1

    count = int(r.read_ubit_var())
    for i in range(0, count):
        fp.last += 1
        fp.path[fp.last] = int(r.read_ubit_var_field_path())


def PopOnePlusOne(r, fp):
    fp.path[fp.last] = 0
    fp.last -= 1
    fp.path[fp.last] += 1


def PopOnePlusN(r, fp):
    fp.path[fp.last] = 0
    fp.last -= 1
    fp.path[fp.last] += int(r.read_ubit_var_field_path()) + 1


def PopAllButOnePlusOne(r, fp):
    pop_num = fp.last
    for i in range(0, pop_num):
        fp.path[fp.last] = 0
        fp.last -= 1

    fp.path[0] += 1


def PopAllButOnePlusN(r, fp):
    pop_num = fp.last
    for i in range(0, pop_num):
        fp.path[fp.last] = 0
        fp.last -= 1

    fp.path[0] += int(r.read_ubit_var_field_path()) + 1


def PopAllButOnePlusNPack3Bits(r, fp):
    pop_num = fp.last
    for i in range(0, pop_num):
        fp.path[fp.last] = 0
        fp.last -= 1

    fp.path[0] += int(r.read_bits(3)) + 1


def PopAllButOnePlusNPack6Bits(r, fp):
    pop_num = fp.last
    for i in range(0, pop_num):
        fp.path[fp.last] = 0
        fp.last -= 1

    fp.path[0] += int(r.read_bits(6)) + 1


def PopNPlusOne(r, fp):
    pop_num = int(r.read_ubit_var_field_path())
    for i in range(0, pop_num):
        fp.path[fp.last] = 0
        fp.last -= 1

    fp.path[fp.last] += 1


def PopNPlusN(r, fp):
    pop_num = int(r.read_ubit_var_field_path())
    for i in range(0, pop_num):
        fp.path[fp.last] = 0
        fp.last -= 1

    fp.path[fp.last] += int(r.read_var_int32())


def PopNAndNonTopographical(r, fp):
    pop_num = int(r.read_ubit_var_field_path())
    for i in range(0, pop_num):
        fp.path[fp.last] = 0
        fp.last -= 1

    for i in range(0, fp.last + 1):
        if r.read_boolean():
            fp.path[i] += int(r.read_var_int32())


def NonTopoComplex(r, fp):
    for i in range(0, fp.last + 1):
        if r.read_boolean():
            fp.path[i] += int(r.read_var_int32())


def NonTopoPenultimatePlusOne(r, fp):
    fp.path[fp.last - 1] += 1


def NonTopoComplexPack4Bits(r, fp):
    for i in range(0, fp.last + 1):
        if r.read_boolean():
            fp.path[i] += int(r.read_bits(4)) - 7


def FieldPathEncodeFinish(r, fp):
    fp.done = True


EntityOp = NewType('EntityOp', int)
EntityOpNone = EntityOp(0x00)
EntityOpCreated = EntityOp(0x01)
EntityOpUpdated = EntityOp(0x02)
EntityOpDeleted = EntityOp(0x04)
EntityOpEntered = EntityOp(0x08)
EntityOpLeft = EntityOp(0x10)
EntityOpCreatedEntered = EntityOp(EntityOpCreated | EntityOpEntered)
EntityOpUpdatedEntered = EntityOp(EntityOpUpdated | EntityOpEntered)
EntityOpDeletedLeft = EntityOp(EntityOpDeleted | EntityOpLeft)

entityOpNames = {
                 EntityOpNone:           "None",
                 EntityOpCreated:        "Created",
                 EntityOpUpdated:        "Updated",
                 EntityOpDeleted:        "Deleted",
                 EntityOpEntered:        "Entered",
                 EntityOpLeft:           "Left",
                 EntityOpCreatedEntered: "Created+Entered",
                 EntityOpUpdatedEntered: "Updated+Entered",
                 EntityOpDeletedLeft:    "Deleted+Left",
                }

FieldPathTable = [FieldPathOp("PlusOne", 36271, PlusOne),
                  FieldPathOp("PlusTwo", 10334, PlusTwo),
                  FieldPathOp("PlusThree", 1375, PlusThree),
                  FieldPathOp("PlusFour", 646, PlusFour),
                  FieldPathOp("PlusN", 4128, PlusN),
                  FieldPathOp("PushOneLeftDeltaZeroRightZero", 35, PushOneLeftDeltaZeroRightZero),
                  FieldPathOp("PushOneLeftDeltaZeroRightNonZero", 3, PushOneLeftDeltaZeroRightNonZero),
                  FieldPathOp("PushOneLeftDeltaOneRightZero", 521, PushOneLeftDeltaOneRightZero),
                  FieldPathOp("PushOneLeftDeltaOneRightNonZero", 2942, PushOneLeftDeltaOneRightNonZero),
                  FieldPathOp("PushOneLeftDeltaNRightZero", 560, PushOneLeftDeltaNRightZero),
                  FieldPathOp("PushOneLeftDeltaNRightNonZero", 471, PushOneLeftDeltaNRightNonZero),
                  FieldPathOp("PushOneLeftDeltaNRightNonZeroPack6Bits", 10530, PushOneLeftDeltaNRightNonZeroPack6Bits),
                  FieldPathOp("PushOneLeftDeltaNRightNonZeroPack8Bits", 251, PushOneLeftDeltaNRightNonZeroPack8Bits),
                  FieldPathOp("PushTwoLeftDeltaZero", 0, PushTwoLeftDeltaZero),
                  FieldPathOp("PushTwoPack5LeftDeltaZero", 0, PushTwoPack5LeftDeltaZero),
                  FieldPathOp("PushThreeLeftDeltaZero", 0, PushThreeLeftDeltaZero),
                  FieldPathOp("PushThreePack5LeftDeltaZero", 0, PushThreePack5LeftDeltaZero),
                  FieldPathOp("PushTwoLeftDeltaOne", 0, PushTwoLeftDeltaOne),
                  FieldPathOp("PushTwoPack5LeftDeltaOne", 0, PushTwoPack5LeftDeltaOne),
                  FieldPathOp("PushThreeLeftDeltaOne", 0, PushThreeLeftDeltaOne),
                  FieldPathOp("PushThreePack5LeftDeltaOne", 0, PushThreePack5LeftDeltaOne),
                  FieldPathOp("PushTwoLeftDeltaN", 0, PushTwoLeftDeltaN),
                  FieldPathOp("PushTwoPack5LeftDeltaN", 0, PushTwoPack5LeftDeltaN),
                  FieldPathOp("PushThreeLeftDeltaN", 0, PushThreeLeftDeltaN),
                  FieldPathOp("PushThreePack5LeftDeltaN", 0, PushThreePack5LeftDeltaN),
                  FieldPathOp("PushN", 0, PushN),
                  FieldPathOp("PushNAndNonTopological", 310, PushNAndNonTopological),
                  FieldPathOp("PopOnePlusOne", 2, PopOnePlusOne),
                  FieldPathOp("PopOnePlusN", 0, PopOnePlusN),
                  FieldPathOp("PopAllButOnePlusOne", 1837, PopAllButOnePlusOne),
                  FieldPathOp("PopAllButOnePlusN", 149, PopAllButOnePlusN),
                  FieldPathOp("PopAllButOnePlusNPack3Bits", 300, PopAllButOnePlusNPack3Bits),
                  FieldPathOp("PopAllButOnePlusNPack6Bits", 634, PopAllButOnePlusNPack6Bits),
                  FieldPathOp("PopNPlusOne", 0, PopNPlusOne),
                  FieldPathOp("PopNPlusN", 0, PopNPlusN),
                  FieldPathOp("PopNAndNonTopographical", 1, PopNAndNonTopographical),
                  FieldPathOp("NonTopoComplex", 76, NonTopoComplex),
                  FieldPathOp("NonTopoPenultimatePlusOne", 271, NonTopoPenultimatePlusOne),
                  FieldPathOp("NonTopoComplexPack4Bits", 99, NonTopoComplexPack4Bits),
                  FieldPathOp("FieldPathEncodeFinish", 25474, FieldPathEncodeFinish)
                 ]


class HuffmanLeaf:
    def __init__(self, weight, value):
        self.weight = weight
        self.value = value


class huffmanNode:
    def __init__(self, weight, value, left, right):
        self.weight = weight
        self.value = value
        self.left  = left
        self.right = right


def Heapify(arr, n, i):
    largest = i  # Initialize largest as root
    l = 2 * i + 1     # left = 2*i + 1
    r = 2 * i + 2     # right = 2*i + 2
  
    # See if left child of root exists and is
    # greater than root
    if l < n and arr[i].weight == arr[l].weight:
        if arr[i].value >= arr[l].value:
            largest = l
    elif l < n and arr[i].weight < arr[l].weight:
        largest = l
  
    # See if right child of root exists and is
    # greater than root
    if r < n and arr[largest].weight == arr[r].weight:
        if arr[largest].value >= arr[r].value:
            largest = r
    elif r < n and arr[largest].weight < arr[r].weight:
        largest = r
  
    # Change root, if needed
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]  # swap
  
        # Heapify the root.
        Heapify(arr, n, largest)
  

# The main function to sort an array of given size
def HeapSort(arr):
    n = len(arr)
  
    # Build a maxheap.
    # Since last parent will be at ((n//2)-1) we can start at that location.
    for i in range(n // 2 - 1, -1, -1):
        Heapify(arr, n, i)
  
    # One by one extract elements
    for i in range(n-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]   # swap
        Heapify(arr, i, 0)


def ReadFieldPaths(r):
    trees = []
    for i, field_path_op in enumerate(FieldPathTable):
        if field_path_op.weight == 0:
            leaf = HuffmanLeaf(1, i)
        else:
            leaf = HuffmanLeaf(field_path_op.weight, i)

        trees.append(leaf)

    HeapSort(trees)
    n = len(trees)
    for i in range(0, n - 1):
        a = trees.pop(0)
        b = trees.pop(0)
        new_tree = huffmanNode(a.weight + b.weight, i + 40, a, b)
        
        trees.append(new_tree)
        HeapSort(trees)

    fp = FieldPath()

    paths = []
    node_tree, next_tree = trees[0], trees[0]
    while not fp.done:
        flag = r.read_bits(1)
        if flag == 1:
            next_tree = node_tree.right
        else:
            next_tree = node_tree.left

        if type(next_tree) == type(HuffmanLeaf(0, 0)):
            node_tree = trees[0]
            next_value = next_tree.value

            #print("next_value: ", next_value)
            FieldPathTable[next_value].fn(r, fp)
            #print("fp.path: ", fp.path)
            if not fp.done:
                paths.append(copy.deepcopy(fp))
        else:
            node_tree = next_tree

    return paths


def GetDecoderForFieldPathFromSerializer(s, fp, pos):
    index = fp.path[pos]
    field = s.fields[index]
    decoder, f = GetDecoderForFieldPathFromField(field, fp, pos + 1)

    return decoder, f


def GetDecoderForFieldPathFromField(f, fp, pos):
    if f.model == fieldModelEnum.fieldModelFixedArray.value:
        decoder = f.decoder

        return decoder, f
    elif f.model == fieldModelEnum.fieldModelFixedTable.value:
        if fp.last ==  pos - 1:
            decoder = f.base_decoder

            return decoder, f
        else:
            decoder, f = GetDecoderForFieldPathFromSerializer(f.serializer, fp, pos)

            return decoder, f

    elif f.model == fieldModelEnum.fieldModelVariableArray.value:
        if fp.last == pos:
            decoder = f.child_decoder
        else:
            decoder = f.base_decoder

        return decoder, f
    elif f.model == fieldModelEnum.fieldModelVariableTable.value:
        if fp.last >= pos + 1:
            decoder, f = GetDecoderForFieldPathFromSerializer(f.serializer, fp, pos + 1)
            return decoder, f
        else:
            decoder = f.base_decoder

            return decoder, f
    else:
        decoder = f.decoder

        return decoder, f


def ReadFields(r, s, state):
    paths = ReadFieldPaths(r)
    path_list = [path.path for path in paths]
    for path in paths:
        decoder, field = GetDecoderForFieldPathFromSerializer(s, path, 0)

        #print("decoder: ", decoder)

        val = None
        if decoder == "signedDecoder":
            n = r.bit_count
            val = r.read_var_int32()
        elif decoder == "noscaleDecoder":
            val = r.read_bits(32)
            val = int(bin(val), 2)
            val = struct.unpack('f', struct.pack('I', val))[0]
            val = round(val, 1)
        elif decoder == "unsignedDecoder":
            val = r.read_var_uint32()
        elif decoder == "booleanDecoder":
            val = r.read_boolean()
        elif decoder == "floatCoordDecoder_3":
            val = [None, None, None]
            for i in range(3):
                val[i] = r.read_coord()

        elif decoder == "QuantizedFloatDecoder":
            qfd = QuantizedFloatDecoder(field.bit_count, field.encode_flags, 
                                        field.low_value, field.high_value)
            val = qfd.decode(r)
            val = round(val, 5)
        elif decoder == "defaultDecoder":
            val = r.read_var_uint32()
        elif decoder == "stringDecoder":
            val = r.read_string()
        elif decoder == "noscaleDecoder_3":
            val = [None, None, None]
            for i in range(3):
                value = r.read_bits(32)
                value = int(bin(value), 2)
                value = struct.unpack('f', struct.pack('I', value))[0]
                value = round(value, 1)
                val[i] = value

        elif decoder == "unsigned64Decoder":
            val = r.read_var_uint64()
        elif decoder == "simulationTimeDecoder":
            val = np.float32(r.read_var_uint32()) * (1.0 / 30)
        elif decoder == "floatCoordDecoder":
            val = r.read_coord()
        elif decoder == "QAngle_1":
            val = [0, 0, 0]
            n = field.bit_count

            val[0] = r.read_angle(n)
            val[1] = r.read_angle(n)
            val[2] = 0.0
        elif decoder == "QAngle_3":
            val = [0, 0, 0]

            rX = r.read_boolean()
            rY = r.read_boolean()
            rZ = r.read_boolean()
            if rX:
                val[0] = r.read_coord()

            if rY:
                val[1] = r.read_coord()

            if rZ:
                val[2] = r.read_coord()    
        elif decoder == "vectorNormalDecoder":
            val = r.read_3bit_normal()
        elif decoder == "fixed64Decoder":
            val = r.read_le_uint64()

        #if val == 64576.31158:
        #if decoder == "QuantizedFloatDecoder":
            #print("decoder: ", decoder)
            #print("val: ", val)
            #print("")

        SetFieldState(state, path, val)


class DOTA_COMBATLOG_TYPES(Enum):
    DOTA_COMBATLOG_INVALID = -1;
    DOTA_COMBATLOG_DAMAGE = 0;
    DOTA_COMBATLOG_HEAL = 1;
    DOTA_COMBATLOG_MODIFIER_ADD = 2;
    DOTA_COMBATLOG_MODIFIER_REMOVE = 3;
    DOTA_COMBATLOG_DEATH = 4;
    DOTA_COMBATLOG_ABILITY = 5;
    DOTA_COMBATLOG_ITEM = 6;
    DOTA_COMBATLOG_LOCATION = 7;
    DOTA_COMBATLOG_GOLD = 8;
    DOTA_COMBATLOG_GAME_STATE = 9;
    DOTA_COMBATLOG_XP = 10;
    DOTA_COMBATLOG_PURCHASE = 11;
    DOTA_COMBATLOG_BUYBACK = 12;
    DOTA_COMBATLOG_ABILITY_TRIGGER = 13;
    DOTA_COMBATLOG_PLAYERSTATS = 14;
    DOTA_COMBATLOG_MULTIKILL = 15;
    DOTA_COMBATLOG_KILLSTREAK = 16;
    DOTA_COMBATLOG_TEAM_BUILDING_KILL = 17;
    DOTA_COMBATLOG_FIRST_BLOOD = 18;
    DOTA_COMBATLOG_MODIFIER_STACK_EVENT = 19;
    DOTA_COMBATLOG_NEUTRAL_CAMP_STACK = 20;
    DOTA_COMBATLOG_PICKUP_RUNE = 21;
    DOTA_COMBATLOG_REVEALED_INVISIBLE = 22;
    DOTA_COMBATLOG_HERO_SAVED = 23;
    DOTA_COMBATLOG_MANA_RESTORED = 24;
    DOTA_COMBATLOG_HERO_LEVELUP = 25;
    DOTA_COMBATLOG_BOTTLE_HEAL_ALLY = 26;
    DOTA_COMBATLOG_ENDGAME_STATS = 27;
    DOTA_COMBATLOG_INTERRUPT_CHANNEL = 28;
    DOTA_COMBATLOG_ALLIED_GOLD = 29;
    DOTA_COMBATLOG_AEGIS_TAKEN = 30;
    DOTA_COMBATLOG_MANA_DAMAGE = 31;
    DOTA_COMBATLOG_PHYSICAL_DAMAGE_PREVENTED = 32;
    DOTA_COMBATLOG_UNIT_SUMMONED = 33;
    DOTA_COMBATLOG_ATTACK_EVADE = 34;
    DOTA_COMBATLOG_TREE_CUT = 35;
    DOTA_COMBATLOG_SUCCESSFUL_SCAN = 36;
    DOTA_COMBATLOG_END_KILLSTREAK = 37;
    DOTA_COMBATLOG_BLOODSTONE_CHARGE = 38;
    DOTA_COMBATLOG_CRITICAL_DAMAGE = 39;
    DOTA_COMBATLOG_SPELL_ABSORB = 40;
    DOTA_COMBATLOG_UNIT_TELEPORTED = 41;
    DOTA_COMBATLOG_KILL_EATER_EVENT = 42;


class DOTA_UNIT_ORDER_TYEPS(Enum):
    DOTA_UNIT_ORDER_NONE = 0;
    DOTA_UNIT_ORDER_MOVE_TO_POSITION = 1;
    DOTA_UNIT_ORDER_MOVE_TO_TARGET = 2;
    DOTA_UNIT_ORDER_ATTACK_MOVE = 3;
    DOTA_UNIT_ORDER_ATTACK_TARGET = 4;
    DOTA_UNIT_ORDER_CAST_POSITION = 5;
    DOTA_UNIT_ORDER_CAST_TARGET = 6;
    DOTA_UNIT_ORDER_CAST_TARGET_TREE = 7;
    DOTA_UNIT_ORDER_CAST_NO_TARGET = 8;
    DOTA_UNIT_ORDER_CAST_TOGGLE = 9;
    DOTA_UNIT_ORDER_HOLD_POSITION = 10;
    DOTA_UNIT_ORDER_TRAIN_ABILITY = 11;
    DOTA_UNIT_ORDER_DROP_ITEM = 12;
    DOTA_UNIT_ORDER_GIVE_ITEM = 13;
    DOTA_UNIT_ORDER_PICKUP_ITEM = 14;
    DOTA_UNIT_ORDER_PICKUP_RUNE = 15;
    DOTA_UNIT_ORDER_PURCHASE_ITEM = 16;
    DOTA_UNIT_ORDER_SELL_ITEM = 17;
    DOTA_UNIT_ORDER_DISASSEMBLE_ITEM = 18;
    DOTA_UNIT_ORDER_MOVE_ITEM = 19;
    DOTA_UNIT_ORDER_CAST_TOGGLE_AUTO = 20;
    DOTA_UNIT_ORDER_STOP = 21;
    DOTA_UNIT_ORDER_TAUNT = 22;
    DOTA_UNIT_ORDER_BUYBACK = 23;
    DOTA_UNIT_ORDER_GLYPH = 24;
    DOTA_UNIT_ORDER_EJECT_ITEM_FROM_STASH = 25;
    DOTA_UNIT_ORDER_CAST_RUNE = 26;
    DOTA_UNIT_ORDER_PING_ABILITY = 27;
    DOTA_UNIT_ORDER_MOVE_TO_DIRECTION = 28;
    DOTA_UNIT_ORDER_PATROL = 29;
    DOTA_UNIT_ORDER_VECTOR_TARGET_POSITION = 30;
    DOTA_UNIT_ORDER_RADAR = 31;
    DOTA_UNIT_ORDER_SET_ITEM_COMBINE_LOCK = 32;
    DOTA_UNIT_ORDER_CONTINUE = 33;
    DOTA_UNIT_ORDER_VECTOR_TARGET_CANCELED = 34;
    DOTA_UNIT_ORDER_CAST_RIVER_PAINT = 35;
    DOTA_UNIT_ORDER_PREGAME_ADJUST_ITEM_ASSIGNMENT = 36;
    DOTA_UNIT_ORDER_DROP_ITEM_AT_FOUNTAIN = 37;
    DOTA_UNIT_ORDER_TAKE_ITEM_FROM_NEUTRAL_ITEM_STASH = 38;
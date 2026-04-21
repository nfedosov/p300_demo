# -*- coding: utf-8 -*-
import numpy as np
from pylsl import StreamInlet, resolve_byprop
from pylsl.pylsl import lib, StreamInfo, FOREVER, c_int, c_double, byref, handle_error
import time
#import socket
import xml.etree.ElementTree as ET
LSL_STREAM_NAMES = ['NVX52_Data']
LSL_RESOLVE_TIMEOUT = 2



class FixedStreamInfo(StreamInfo):
    def as_xml(self):
        return lib.lsl_get_xml(self.obj).decode('utf-8', 'ignore') # add ignore


class FixedStreamInlet(StreamInlet):
    def info(self, timeout=FOREVER):
        errcode = c_int()
        result = lib.lsl_get_fullinfo(self.obj, c_double(timeout),
                                      byref(errcode))
        handle_error(errcode)
        return FixedStreamInfo(handle=result) # StreamInfo(handle=result)

# TODO: make default resolving names and choosing the first stream, dive into the details

class LSLInlet:
    def __init__(self, params, dtype='float'):
        self.params = params
        if self.params['lsl_stream_name']:
            name = self.params['lsl_stream_name']
        else:
            name = LSL_STREAM_NAMES[0]
        streams = resolve_byprop('name', name, timeout=LSL_RESOLVE_TIMEOUT)
        self.inlet = None
        self.dtype = dtype
        if len(streams) > 0:
            self.inlet = FixedStreamInlet(streams[0], max_buflen=params['max_buflen'],
                                          max_chunklen=params['max_chunklen'])  # ??? Check timing!!!
            print('Connected to {} LSL stream successfully'.format(name))
            self.n_channels = self.inlet.info().channel_count()
        else:
            raise ConnectionError('Cannot connect to "{}" LSL stream'.format(name))

    def get_next_chunk(self):
        # get next chunk
        chunk, timestamp = self.inlet.pull_chunk(max_samples=1000*120)
        # convert to numpy array
        chunk = np.array(chunk, dtype=self.dtype)
        # return first n_channels channels or None if empty chunk
        return (chunk, timestamp) if chunk.shape[0] > 0 else (None, None)

  

    def save_info(self, file):
        with open(file, 'w', encoding="utf-8") as f:
            f.write(self.info_as_xml())

    def info_as_xml(self):
        xml = self.inlet.info().as_xml()
        return xml

    def get_frequency(self):
        self.srate = self.inlet.info().nominal_srate()
        return self.srate

    def get_n_channels(self):
        return self.inlet.info().channel_count()

    def get_channel_names(self):
        xml_info = self.info_as_xml()
        tree = ET.ElementTree(ET.fromstring(xml_info))
        root = tree.getroot()
        for child in root:
            print(child.tag, child.attrib)
        return 'puf-puf'




    #  Research this code
    def get_channels_labels(self):
        for t in range(3):
            time.sleep(0.5*(t+1))
            try:
                # print('wow') TODO too many repetitions
                rt = ET.fromstring(self.info_as_xml())
                channels_tree = rt.find('desc').findall("channel") or rt.find('desc').find("channels").findall(
                    "channel")
                labels = [(ch.find('label') if ch.find('label') is not None else ch.find('name')).text
                          for ch in channels_tree]
                return labels
            except OSError:
                print('OSError during reading channels names', t+1)
        return ['channel'+str(n+1) for n in range(self.get_n_channels())]

    def disconnect(self):
        del self.inlet
        self.inlet = None

from PyQt5.QtWidgets import QApplication, QMainWindow, QHBoxLayout,\
    QVBoxLayout, QWidget, QLabel, QListWidget, QListWidgetItem,\
        QFileDialog, QComboBox
from PyQt5.QtCore import Qt, QMimeData, pyqtSignal, QRect, QCoreApplication, QPoint
from PyQt5.QtGui import QDrag, QPixmap, QFont,QKeyEvent
from PyQt5 import QtCore, QtGui, QtWidgets
import pickle


from scipy.special import softmax
from scipy.signal import deconvolve
import mne

import matplotlib.pyplot as plt


from PyQt5.QtGui import QPainter, QBrush, QPen, QPolygon

import sys

import numpy as np
from datetime import datetime
import os
from lsl_inlet import LSLInlet
import pylsl

import asyncio
import nest_asyncio
import AsyncKandinsky as kandinsky
import random


from collections import Counter
from heapq import heappush, heappop






def create_balanced_list(elements):
    # Count the frequency of each element
    freq = Counter(elements)
    # Use a max heap (negative counts for max-heap behavior in Python)
    max_heap = []
    for element, count in freq.items():
        heappush(max_heap, (-count, element))
    
    result = []
    prev_count, prev_elem = 0, None  # Track the previous element used
    
    while max_heap:
        count, elem = heappop(max_heap)  # Get the most frequent element
        result.append(elem)
        
        # Push the previous element back if its count is still non-zero
        if prev_count < 0:
            heappush(max_heap, (prev_count, prev_elem))
        
        # Update the previous element and its count
        prev_count = count + 1  # Decrease the count since it was used
        prev_elem = elem

    # If the result size doesn't match the input size, balancing isn't possible
    if len(result) != len(elements):
        raise ValueError("Cannot create a balanced list with the given elements.")
    
    return result



class ClosableLabel(QtWidgets.QLabel):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setAlignment(QtCore.Qt.AlignCenter)  # Center align the text
        #self.setStyleSheet("font-size: 20px; background-color: lightblue; padding: 10px;")
    
    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == QtCore.Qt.Key_Space:
            self.close()  # Close the label
        else:
            super().keyPressEvent(event)  # Handle other keys normally




class ShapeLabel(QLabel):
    def __init__(self, parent=None):
        super(ShapeLabel, self).__init__(parent)
        self.shape_params = None   
    def setImage(self, kind,center_x = None, center_y= None, width= None, height= None, color=Qt.black, stim = 0):
        self.shape_params = (kind,center_x, center_y, width, height, color,stim)
        if kind != 'clear':
            pic_name = kind
            pixmap = QPixmap(pic_name)
            self.setPixmap(pixmap)
        self.update()
        

    def paintEvent(self, event):
        super(ShapeLabel, self).paintEvent(event)
        if self.shape_params:
            kind,center_x, center_y, width, height, color,stim = self.shape_params
            painter = QPainter(self)
            
            #if stim == 0:
            #    painter.setPen(QPen(Qt.gray, 2, Qt.SolidLine))
            #    painter.setBrush(QBrush(Qt.gray, Qt.SolidPattern))
                
            #if stim == 1:
            #    painter.setPen(QPen(Qt.gray, 2, Qt.SolidLine))
            #    painter.setBrush(QBrush(Qt.gray, Qt.SolidPattern))

            #painter.drawRect(self.width()-200, 0, self.width(),200)

            #painter.setPen(QPen(color, 20))# Qt.DashLine))
            #painter.setBrush(Qt.NoBrush)

            if kind == 'clear':
                return


class ProtocolBlock:
    def __init__(self, name, duration,message,code):
        self.name = name
        self.duration = duration
        self.message = message
        self.code = code


class ProtocolEditor(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.protocol_blocks = []
        self.init_ui()

    def init_ui(self):
        self.setAcceptDrops(True)
        self.layout = QtWidgets.QVBoxLayout()
           
        self.path_to_pics = 'elements'
        

        self.all_stages_scenes = [['candy','bear','book','guitar','skis']]

        #self.all_stages_scenes_rus_instr = [['Летний лес','Осенний лес','Горы','Пляж','Город','Поле']]

        self.colors =  ['blue','orange','green','violet','red','lightblue','yellow','navy']

        #self.all_stages_scenes_rus_last = [['летнем лесе','осеннем лесе','горах','пляже','городе','поле']]


        self.shape_L = 600
        
        self.stim_mode = 'comp'
        
        self.setLayout(self.layout)

        self.lsl_button = QtWidgets.QPushButton('UPD lsl streams')
        self.lsl_button.setStyleSheet("""
    QPushButton {
        background-color: lightblue; 
        color: black;
        border-radius: 5px;
        padding: 10px;
        font: bold 30px "castellar";
    }
    

    QPushButton:hover {
        background-color: orange;
    }
""")
        self.lsl_button.clicked.connect(self.upd_lsl_streams)

        self.lsl_combobox = QComboBox(self)
        self.lsl_combobox.setStyleSheet("""
    QComboBox {
        background-color: silver; 
        color: black;
        border-radius: 5px;
        padding: 15px;
        font: 20px "gost";
    }
    

    QComboBox:hover {
        background-color: orange;
    }
""")
        
        self.lsl_combobox.ReadOnly = True
        self.lsl_combobox.currentIndexChanged.connect(self.choose_lsl)

        self.start_button = QtWidgets.QPushButton('Start')
        self.start_button.setStyleSheet("""
    QPushButton {
        background-color: springgreen; 
        color: black;
        border-radius: 5px;
        padding: 10px;
        font: bold 30px "castellar";
    }
    

    QPushButton:hover {
        background-color: orange;
    }
""")
        
        self.start_button.clicked.connect(self.onStartButtonClicked)
        
        
        # Patch the running event loop
        nest_asyncio.apply()

        # Initialize the model
        self.model = kandinsky.FusionBrainApi(kandinsky.ApiWeb("np_fedosov@list.ru", "2086schA1"))

    #async def generate(self,category):
    #    try:
    #        result = await self.model.text2image(category, style="DEFAULT", art_gpt=True)
    #        # Новый параметр art_gpt - это инструмент для автоматического улучшения промпта => улучшение качества картинки
    #    except ValueError as e:
    #        print(f"Error:\t{e}")
    #    else:
    #        # Save the generated image
    #        with open("forest.png", "wb") as f:
    #            f.write(result.getvalue())
    #        print("Done!")

    #async def call_generation(self, n):
    #    """
    #    Calls the `generate` function `n` times with random categories from the stimuli list.
    #    """
    #    tasks = []
    #    for _ in range(n):
    #        category = random.choice(self.list_of_stimuli)
    #        
    #        result = self.generate(category)
    #        with open("archive"+str(_)+".png", "wb") as f:
    #            f.write(result.getvalue())
    #        print("Done!")
    #    await asyncio.gather(*tasks)

          
    def upd_lsl_streams(self):
        """Query available LSL streams and update stream selection combobox."""
        self.streams = pylsl.resolve_streams()
        
        for stream in self.streams:
            self.lsl_combobox.addItem(stream.name())
        
        if self.streams:
            self.inlet_info = self.streams[0]
                    
            
    def choose_lsl(self, idx):
        
        self.inlet_info = self.streams[idx]
        print(self.inlet_info)
        
    def onRefreshImages(self):
        pass
    
        # start generating images untill nutton is released

    def onStartButtonClicked(self):
        
        
        

        self.stim_label = ShapeLabel('+')
        
        #instr_text = 'Представьте, что вы могли могли бы отдохнуть\nв одной из нижеперечисленных локаций:\n\n\nЛетний лес\t\tОсеннний лес\t\tГоры\t\tПоле\t\tПляж\t\tГород\n\n\nЗагадайте место, в котором вы бы хотели оказаться - считайте\nкаждый раз, когда вы видете соответствующую картинку,\nа мы попробуем прочитать ваши мысли.\nСмотрите в центр экрана, не двигайтесь и не разговаривайте\nво время прохождения эксперимента\n'
        instr_text = 'Какой подарок вы бы хотели получить на Новый Год?\nЗагадайте что-то из списка:\n\n\nКнига\t\tПлюшевый медведь\t\tКоробка конфет\t\tГитара\t\tКоньки\n\n\nЗагадайте предмет, который вы хотели бы получить, \nи ожидайте предъявления соответствующей картинки,\nа мы попробуем прочитать ваши мысли.\nЧтобы желание точно сбылось, считайте про себя каждый раз,\nкогда видите загаданный предмет\nСмотрите в центр экрана, не двигайтесь и не разговаривайте\nво время прохождения эксперимента\n'
        
        self.instr_label = ClosableLabel(instr_text)
        self.instr_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self.instr_label.setAlignment(Qt.AlignCenter)
        
        font = QFont( "Gost", 26)# QFont.Bold)
        self.instr_label.setFont(font)
        
        self.instr_label.showMaximized() 
        
        
        while(self.instr_label.isVisible()):
            QCoreApplication.processEvents()
    
            
        

        self.stim_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self.stim_label.setAlignment(Qt.AlignCenter)
        
        
        font = QFont( "Gost", 50)# QFont.Bold)
        self.stim_label.setFont(font)
        
        
        
        
        
        self.stim_label.showMaximized() 
        
        
        n_samples_received_prev = 0#??
        n_samples_received = 0



        for st in range(len(self.all_stages_scenes)):
    
            self.protocol_blocks = []
        
            pb_pause = ProtocolBlock('PAUSE',1,'+',-1)
            pb_pause2 = ProtocolBlock('non_pause',3,'+',-2)
            
            
            
            #instruction = ProtocolBlock('instruction',10,instr_text,-3)
            
            
            self.list_of_scenes = self.all_stages_scenes[st]#['triangle','square','circle']
    
            self.list_of_names = []
            
    
            for count,scene in enumerate(self.list_of_scenes):
                self.list_of_names.append([])
                for filename in os.listdir(os.path.join(self.path_to_pics,scene)):
    
                    self.list_of_names[count].append(os.path.join(self.path_to_pics,scene,filename))
    
            self.list_of_names_gift = []
            for count,scene in enumerate(self.list_of_scenes):
                self.list_of_names_gift.append([])
                for filename in os.listdir(os.path.join(self.path_to_pics,scene+'_gift')):
    
                    self.list_of_names_gift[count].append(os.path.join(self.path_to_pics,scene+'_gift',filename))
    
    
    
            self.score = np.zeros((len(self.list_of_scenes),))      
            self.probabilities = np.ones((len(self.list_of_scenes),))/len(self.list_of_scenes)
            self.labels_prob = self.list_of_scenes.copy()
            
            N_single_stims = 80
     
            N_shapes = len(self.list_of_scenes)
            
    
            base_photo_duration = 0.05
            
            
            
            
            base_shape_name_list = list()
            
            self.protocol_blocks=  list()
            for j in range(N_shapes):
                self.protocol_blocks.append(ProtocolBlock(self.list_of_scenes[j],0.4,'',j))
            
            self.protocol_blocks = self.protocol_blocks*N_single_stims
        
            np.random.shuffle(self.protocol_blocks)#create_balanced_list(base_shape_name_list) 
            
            
           
       
            '''
            base_shape_name_list = list()
            
            
          
            
            for j in range(N_shapes):

                base_shape_name_list.append(self.list_of_scenes[j])
                
            base_shape_name_list = base_shape_name_list*N_single_stims
            #for k in range(N_single_stims):
            block_list_names = base_shape_name_list.copy()
            np.random.shuffle(block_list_names)#create_balanced_list(base_shape_name_list) 
            
            
            self.protocol_blocks=  list()
            for k in range(len(block_list_names)):
                self.protocol_blocks.append(ProtocolBlock(block_list_names[k],0.8,'',k))
            '''
     
            
            self.protocol_blocks.insert(0,pb_pause2)
            
            self.protocol_blocks.append(pb_pause)
            
            
            #self.protocol_blocks.insert(0,instruction)
            #self.protocol_blocks.append(pb_pause)
             

            #self.protocol_blocks.append(pb_pause2)
            #self.protocol_blocks.append(pb_pause)
 
            blocks = dict()
            for i in range(len(self.protocol_blocks)):
                if self.protocol_blocks[i].name not in blocks:
                    blocks[self.protocol_blocks[i].name] = {'duration': self.protocol_blocks[i].duration, 'id': self.protocol_blocks[i].code, 'message': self.protocol_blocks[i].message}
      
            seq2 = list()
            for i in range(len(self.protocol_blocks)):
                seq2.append(self.protocol_blocks[i].name)
      
            timestamp_str = datetime.strftime(datetime.now(), '%m-%d_%H-%M-%S')
            results_path = 'results/{}_{}/'.format('baseline_experiment', timestamp_str)
            #colors =  ['blue','orange','green','violet','red','lightblue','yellow','navy']
    
            
            exp_settings = {
                        'exp_name': 'Baseline',
                        'lsl_stream_name': self.inlet_info.name(),
                        'blocks': blocks,
                        'sequence': seq2,
            
                        #максимальная буферизация принятых данных через lsl
                        'max_buflen': 5,  # in seconds!!!!
            
                        #максимальное число принятых семплов в чанке, после которых появляется возможность
                        #считать данные
                        'max_chunklen': 1, 
    
                        'results_path': results_path}
    
            inlet = LSLInlet(exp_settings)
            inlet.srate = inlet.get_frequency()
            print(inlet.srate)
            xml_info = inlet.info_as_xml()
            channel_names = inlet.get_channels_labels()
            print(channel_names)
            exp_settings['channel_names'] = channel_names
            n_channels = len(channel_names)
    
            srate = int(round(inlet.srate)) #Hz
            exp_settings['srate'] = srate
    
            total_samples = 0
            for block_name in exp_settings['sequence']:
                current_block = exp_settings['blocks'][block_name]
                total_samples += round(srate * current_block['duration'])
            data = np.zeros((total_samples+round(exp_settings['max_buflen']*srate),n_channels))
            
            data[:,-1] -= 5
            print(data.shape)
    
            
            n_samples_received_in_block = 0
    
            block_idx = 0
            block_name = exp_settings['sequence'][0]
            current_block = exp_settings['blocks'][block_name]
            n_samples = srate * current_block['duration']
            block_id = current_block['id']
      
            self.stim_label.clear()  
            self.stim_label.setImage('clear',None,stim = 0)
            self.stim_label.setText(current_block['message'])
          
            QCoreApplication.processEvents()
    
            prev_lab = 0
            
            
            N_counter = 0
            prev_block_idx = 0
            
            
            if self.stim_mode == 'comp':
                bool_stims = np.zeros((total_samples+round(exp_settings['max_buflen']*srate),),dtype = 'int')
                
            
            
            
            
            while (1):
                if n_samples_received_in_block >= n_samples:
                    dif =  n_samples_received_in_block - n_samples
                    
                    
                    block_idx += 1
                    
                    
                    if block_idx >= len(exp_settings['sequence']):
                        #save_and_finish()
                        inlet.disconnect()
            
                        break
            
                    print(block_name)
                    block_name = exp_settings['sequence'][block_idx]
                    current_block = exp_settings['blocks'][block_name]
                    
                    if current_block['id']>= 0:
                        num_pic = np.random.randint(0,len(self.list_of_names[current_block['id']]))
                        pth = self.list_of_names[current_block['id']][num_pic]
                        
                        
                        if self.stim_mode == 'comp':
                            bool_stims[n_samples_received] = current_block['id']+1
                        
                        
                        
                        
                    
                    
                    n_samples = srate * current_block['duration']
                    n_samples_received_in_block = dif
                    block_id = current_block['id']
                    
                    
                    
                                          
                    
                             
                    
                if n_samples_received_in_block >= int(srate*(base_photo_duration+current_block['id']*base_photo_duration)):
                    stim = 0
                else:
                    stim = 1
                
                
                self.stim_label.clear()
                self.stim_label.setImage('clear',None,stim = stim)
                
                
                
                #if current_block['name']= 'instruction':
                self.stim_label.setText(current_block['message'])
                #else:
                #    self.
                
                
            
                if current_block['id']>= 0:
                     
                    
                    if self.stim_mode =='comp':
                        self.stim_label.setImage(pth,stim = 0)
                    else:
                        self.stim_label.setImage(pth,stim = stim)
                    
                    
                    
                if current_block['id'] == -1:
                    
                    
                    #if prev_block_idx != block_idx:
                    #    N_counter += 1
                    #    prev_block_idx = block_idx
            
                    
                     
                        eeg = data[:n_samples_received,:].copy()
                        
                        Nch= eeg.shape[1]-1
                        nT = eeg.shape[0]
     
                        
                        
                        
                        
                        
                        info = mne.create_info(exp_settings['channel_names'][:-1], exp_settings['srate'], ch_types='eeg')
                        
                        
                        raw = mne.io.RawArray(eeg.T[:-1,:],info)
                        
                
                        
                        raw.notch_filter([50.0,100.0])
                        

                        
                        raw.filter(0.3,20.0)
                        
                        
                        
                        
                        
                        
                        if self.stim_mode != 'comp':
                        
                            bool_stims = np.zeros(nT,dtype = 'int')
                            
                            
                            
                            
                            
                            stim_data = eeg[:,-1]
             
                            thr = np.mean(stim_data[:n_samples_received])
        
                        
                            
                            
                            
                            bias_from_start = int(1.0*raw.info['sfreq'])
                            detected_stim = False
                            for i in range(bias_from_start,nT):
                                if (stim_data[i] > thr) and not detected_stim:
                                    minicount = 0
                                    sumts = int(base_photo_duration/2*srate)
                                    
                                    while stim_data[i+sumts] >thr:
                                        sumts += int(base_photo_duration*srate)
                                        minicount +=1
                                    
                                    bool_stims[i] = minicount
                  
                                    detected_stim = True          
                                if (stim_data[i] <thr) and detected_stim:
                                    detected_stim = False
        
                        all_stims_stamps = np.where(bool_stims)[0]
        
                        
                        #%%
               
                        
                        descriptions = list()
                        for i in range(len(all_stims_stamps)):
                            descriptions.append(self.list_of_scenes[bool_stims[all_stims_stamps[i]]-1])

                        annotations = mne.Annotations(onset = all_stims_stamps/raw.info['sfreq'], duration = 1.0, description = descriptions)
                            
                        raw.set_annotations(annotations)

                            
                        events = mne.events_from_annotations(raw)
                            
                        
                        
                        baseline_len = -0.1#-0.1
                            
 
                        epoched = mne.Epochs(raw,events[0],events[1],tmin = baseline_len, tmax = 1.0)#,baseline = None)#,reject = reject_criteria,tmin = 0, tmax = 6.0,baseline = None)
                            
                        #xdawn = mne.preprocessing.Xdawn(correct_overlap='auto')
                        #xdawn.fit(epoched)

                        
                      



                        low_bound_dict = [0.25,0.3,0.35,0.4,0.45,0.5]
                        high_bound_dict = [0.4,0.45,0.5,0.55,0.6,0.65]

                        max_score = -1000
                        max_idx = 0
                        
                        for bound_idx in range(len(low_bound_dict)):
                            epochs = []
                            aves = []
                            raw_scores  =[]
                            
                            low_bound = int((low_bound_dict[bound_idx]-baseline_len)*srate)
                            high_bound = int((high_bound_dict[bound_idx]-baseline_len)*srate)
                         
                            for new_count, scene in enumerate(self.list_of_scenes):
                                epochs.append(epoched[scene].get_data()[:,0])
                                aves.append(epochs[new_count].mean(axis = 0))
                                raw_scores.append(np.mean(aves[new_count][low_bound:high_bound]))
                            if np.max(raw_scores)> max_score:
                                max_score = np.max(raw_scores)
                                max_idx = bound_idx

                        low_bound = int((low_bound_dict[max_idx]-baseline_len)*srate)
                        high_bound = int((high_bound_dict[max_idx]-baseline_len)*srate)
                     
                        epochs = []
                        aves = []
                        raw_scores  =[]


                        plt.figure()
                        
                        for new_count, scene in enumerate(self.list_of_scenes):
                            epochs.append(epoched[scene].get_data()[:,0])
                            aves.append(epochs[new_count].mean(axis = 0))
                            raw_scores.append(np.mean(aves[new_count][low_bound:high_bound]))
                            plt.plot(aves[new_count],color = self.colors[new_count])
                        plt.legend(self.all_stages_scenes[0])
                        pic_name = 'averages.png'
                        plt.savefig(pic_name)

                        logits = np.array(raw_scores)*1e6*0.1*(N_counter+1)
                        
                        self.probabilities = softmax(logits)

                        plt.figure(dpi = 150)
                        plt.bar(self.labels_prob,self.probabilities,color =self.colors)
                        
                        
                        plt.ylim([0,1])
                        plt.ylabel('probability')
                        
                        pic_name = 'probabilities.png'
                        plt.savefig(pic_name)
                        
                      
                        
                        
    
                        pixmap = QPixmap(pic_name)
                        
                        font = QFont( "Gost", 40)# QFont.Bold)
                        self.instr_label.setFont(font)
                        
                        self.instr_label.setText('Нажмите пробел, чтобы получить подарок')
                        
                        
                        self.instr_label.showMaximized()               
            
                        while(self.instr_label.isVisible()):
                            QCoreApplication.processEvents()
                            
                            
                        category_gift = np.argmax(self.probabilities)
                            
                        
                        num_pic = np.random.randint(0,len(self.list_of_names_gift[category_gift]))
                        pth = self.list_of_names_gift[category_gift][num_pic]
                        
                        pixmap = QPixmap(pth)
                        self.instr_label.setPixmap(pixmap)
                        self.instr_label.showMaximized()               
            
                        while(self.instr_label.isVisible()):
                            QCoreApplication.processEvents()
                            
                        
                            
                    
                            
                        
                        
                    
                    #self.stim_label.setPixmap(pixmap)
    
                
                QCoreApplication.processEvents()
            
    
            
                chunk, t_stamp = inlet.get_next_chunk()
                if chunk is not None:
                    n_samples_in_chunk = len(chunk)
                    data[n_samples_received:n_samples_received + n_samples_in_chunk, :] = chunk
                    if self.stim_mode =='comp':
                        data[n_samples_received:n_samples_received + n_samples_in_chunk,-1] = stim#1-stim
            
                    n_samples_received_in_block += n_samples_in_chunk
                    n_samples_received += n_samples_in_chunk
                    
                    
        
        
        data = data[:total_samples]

        os.makedirs(results_path)
        
        file = open(results_path + 'data.pickle', "wb")
        pickle.dump({'eeg': data, #'stim': stims,
                'xml_info': xml_info, 'exp_settings': exp_settings}, file = file)
        file.close()
        

        print('Finished')
        
        self.stim_label.close()
        
        #sys.exit()
        
        #self.call_generation()
    

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setFixedSize(520,285)
        self.protocol_editor = ProtocolEditor()
       
        

        container = QWidget()
        glob_layout = QHBoxLayout()

        protocol_layout = QVBoxLayout()
        #protocol_layout.addStretch(1)
        protocol_layout.addWidget(self.protocol_editor)
        
        
        
        protocol_layout.addWidget(self.protocol_editor.start_button)
        
        protocol_layout.addWidget(self.protocol_editor.lsl_button)
        protocol_layout.addWidget(self.protocol_editor.lsl_combobox)
        protocol_layout.addStretch(1)

        
        container.setLayout(protocol_layout)
     
     
     

        self.setCentralWidget(container)


app = QApplication([])
app.setWindowIcon(QtGui.QIcon("mind.png"))
app.setApplicationName("mindReader")
w = MainWindow()
w.show()

app.exec_()
import yaml 
import os 
d = {"MITBIH":{
        "Models":{ 
            "GRU":os.path.abspath("models/gru/best_model_GRU_mitbih_9817.h5") , 
            "RNN":os.path.abspath("models/rnn/BestModel_RNN_mitbih_908.h5"), 
            "BRNN":os.path.abspath("models/brnn/brnn_mitbih.h5"), 
            "LSTM":os.path.abspath("models/lstm/best_model_lstm_mitbih_976.h5"),
            "Inception":os.path.abspath("models/inception/BestModel_Inception_mitbih_9891.h5"),
            "CNN_Res":os.path.abspath("models/cnn_res/cnn_residual_mitbih.h5"),
            "CNN_LSTM":os.path.abspath("models/cnn_lstm/best_clfc_mitbih_9801.h5")},
        "Data":os.path.abspath("data")
    },
    "PTDB":{
        "Models":{
            "GRU":os.path.abspath("models/gru/BestModel_GRU_ptdb_9014.h5") , 
            "RNN":os.path.abspath("models/rnn/BestModel_RNN_ptdb_91.h5"), 
            "BRNN":os.path.abspath("models/brnn/brnn_ptbdb.h5"), 
            "LSTM":os.path.abspath("models/lstm/best_model_lstm_ptbdb_9673.h5"),
            "Inception":os.path.abspath("models/inception/BestModel_Inception_ptdb_7220.h5"),
            "CNN_Res":os.path.abspath("models/cnn_res/cnn_residual_ptbdb.h5"),
            "CNN_LSTM":os.path.abspath("models/cnn_lstm/best_clf_ptbdb_9776_2cnn.h5")},
        "Data":os.path.abspath("data")
    },
    "Optionals":{ 
        "Optional_1":{
            "Optional1_GRU":os.path.abspath("optionals/optional1/optional1_GRU.h5"),
            "Optional1_RNN":os.path.abspath("optionals/optional1/optional1_RNN.h5"),
            "Optional1_LSTM":os.path.abspath("optionals/optional1/optional1_LSTM.h5")},
        "Optional_2":{
            "Optional2_GRU":os.path.abspath("optionals/optional2/optional2_GRU.h5"),
            "Optional2_RNN":os.path.abspath("optionals/optional2/optional2_RNN.h5"),
            "Optional2_LSTM":os.path.abspath("optionals/optional2/optional2_LSTM.h5")},
        "Optional_3":{
            "Optional3_GRU":os.path.abspath("optionals/optional3/optional3_GRU.h5"),
            "Optional3_RNN":os.path.abspath("optionals/optional3/optional3_RNN.h5"),
            "Optional3_LSTM":os.path.abspath("optionals/optional3/optional3_LSTM.h5")}
    }   
}
         
with open('paths.yaml','w') as outfile : 
    yaml.dump(d,outfile,default_flow_style=False)

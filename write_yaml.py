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
    }
}
         
with open('paths.yaml','w') as outfile : 
    yaml.dump(d,outfile,default_flow_style=False)

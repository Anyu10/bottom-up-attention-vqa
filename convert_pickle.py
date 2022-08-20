import pickle


train_ids_file = 'data/train_ids.pkl'   #* indices to image ids
val_ids_file = 'data/val_ids.pkl'       #* indices to image ids


train_img_ids = pickle.load(open(train_ids_file, "r"))
val_img_ids = pickle.load(open(val_ids_file, "r"))
pickle.dump(train_img_ids, open(train_ids_file, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(val_img_ids, open(val_ids_file, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
